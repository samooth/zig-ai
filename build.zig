const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Detectar CUDA
    const cuda_path = b.option([]const u8, "cuda-path", "Path to CUDA installation")
        orelse std.process.getEnvVarOwned(b.allocator, "CUDA_PATH") catch "/usr/local/cuda";

    const has_cuda = blk: {
        const nvcc = b.findProgram(&.{"nvcc"}, &.{cuda_path ++ "/bin"}) catch break :blk false;
        _ = nvcc;
        break :blk true;
    };

    var ptx_output: ?std.Build.LazyPath = null;
    var cubin_output: ?std.Build.LazyPath = null;
    var dequant_ptx: ?std.Build.LazyPath = null;

    if (has_cuda) {
        const nvcc_path = b.findProgram(&.{"nvcc"}, &.{cuda_path ++ "/bin"}) catch unreachable;

        const compile_ptx = b.addSystemCommand(&.{
            nvcc_path,
            "-arch=sm_80", "-code=sm_80,compute_80",
            "-ptx", "-o",
        });
        ptx_output = compile_ptx.addOutputFileArg("flash_attention.ptx");
        compile_ptx.addFileArg(b.path("cuda/flash_attention.cu"));

        const compile_cubin = b.addSystemCommand(&.{
            nvcc_path,
            "-arch=sm_80", "-cubin", "-o",
        });
        cubin_output = compile_cubin.addOutputFileArg("flash_attention_sm80.cubin");
        compile_cubin.addFileArg(b.path("cuda/flash_attention.cu"));

        const compile_dequant = b.addSystemCommand(&.{
            nvcc_path,
            "-arch=sm_80", "-ptx", "-o",
        });
        dequant_ptx = compile_dequant.addOutputFileArg("dequantize_kernels.ptx");
        compile_dequant.addFileArg(b.path("cuda/dequantize_kernels.cu"));
    }

    // === Módulo core ===
    const core_mod = b.addModule("core", .{
        .root_source_file = b.path("src/tensor.zig"),
        .target = target,
        .optimize = optimize,
    });

    // === Módulo matmul ===
    const matmul_mod = b.addModule("matmul", .{
        .root_source_file = b.path("src/matmul/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    matmul_mod.addImport("core", core_mod);

    // === Módulo fa ===
    const fa_mod = b.addModule("fa", .{
        .root_source_file = b.path("src/fa/flash_attention.zig"),
        .target = target,
        .optimize = optimize,
    });
    fa_mod.addImport("core", core_mod);
    fa_mod.addImport("matmul", matmul_mod);

    // === Módulo cudaz stub ===
    const cudaz_mod = b.addModule("cudaz", .{
        .root_source_file = b.path("src/cuda/cudaz_stub.zig"),
    });

    // === Módulo kv_cache ===
    const kv_cache_mod = b.addModule("kv_cache", .{
        .root_source_file = b.path("src/kv_cache.zig"),
        .target = target,
        .optimize = optimize,
    });
    kv_cache_mod.addImport("core", core_mod);
    kv_cache_mod.addImport("cudaz", cudaz_mod);

    // === NUEVO: Módulo norm ===
    const norm_mod = b.addModule("norm", .{
        .root_source_file = b.path("src/transformer/norm.zig"),
        .target = target,
        .optimize = optimize,
    });
    norm_mod.addImport("core", core_mod);

    // === NUEVO: Módulo ffn ===
    const ffn_mod = b.addModule("ffn", .{
        .root_source_file = b.path("src/transformer/ffn.zig"),
        .target = target,
        .optimize = optimize,
    });
    ffn_mod.addImport("core", core_mod);
    ffn_mod.addImport("matmul", matmul_mod);

    // === NUEVO: Módulo embedding ===
    const embedding_mod = b.addModule("embedding", .{
        .root_source_file = b.path("src/transformer/embedding.zig"),
        .target = target,
        .optimize = optimize,
    });
    embedding_mod.addImport("core", core_mod);
    embedding_mod.addImport("matmul", matmul_mod);

    // === NUEVO: Módulo rope ===
    const rope_mod = b.addModule("rope", .{
        .root_source_file = b.path("src/transformer/rope.zig"),
        .target = target,
        .optimize = optimize,
    });
    rope_mod.addImport("core", core_mod);

    // === NUEVO: Módulo gqa ===
    const gqa_mod = b.addModule("gqa", .{
        .root_source_file = b.path("src/transformer/gqa.zig"),
        .target = target,
        .optimize = optimize,
    });
    gqa_mod.addImport("core", core_mod);

    // === NUEVO: Módulo tokenizer ===
    const tokenizer_mod = b.addModule("tokenizer", .{
        .root_source_file = b.path("src/tokenizer/bpe.zig"),
        .target = target,
        .optimize = optimize,
    });

    // === NUEVO: Módulo loader ===
    const loader_mod = b.addModule("loader", .{
        .root_source_file = b.path("src/loader/safetensors.zig"),
        .target = target,
        .optimize = optimize,
    });
    loader_mod.addImport("core", core_mod);

    // === Módulo transformer (ACTUALIZADO) ===
    const transformer_mod = b.addModule("transformer", .{
        .root_source_file = b.path("src/transformer/layer.zig"),
        .target = target,
        .optimize = optimize,
    });
    transformer_mod.addImport("core", core_mod);
    transformer_mod.addImport("matmul", matmul_mod);
    transformer_mod.addImport("fa", fa_mod);
    transformer_mod.addImport("kv_cache", kv_cache_mod);
    transformer_mod.addImport("norm", norm_mod);
    transformer_mod.addImport("ffn", ffn_mod);
    transformer_mod.addImport("rope", rope_mod);
    transformer_mod.addImport("gqa", gqa_mod);
    transformer_mod.addImport("embedding", embedding_mod);

    // === NUEVO: Módulo pipeline ===
    const pipeline_mod = b.addModule("pipeline", .{
        .root_source_file = b.path("src/transformer/pipeline.zig"),
        .target = target,
        .optimize = optimize,
    });
    pipeline_mod.addImport("core", core_mod);
    pipeline_mod.addImport("matmul", matmul_mod);
    pipeline_mod.addImport("fa", fa_mod);
    pipeline_mod.addImport("transformer", transformer_mod);
    pipeline_mod.addImport("kv_cache", kv_cache_mod);
    pipeline_mod.addImport("embedding", embedding_mod);

    // === Ejecutable principal ===
    const exe = b.addExecutable(.{
        .name = "zig-ai-engine",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("core", core_mod);
    exe.root_module.addImport("matmul", matmul_mod);
    exe.root_module.addImport("fa", fa_mod);
    exe.root_module.addImport("transformer", transformer_mod);
    exe.root_module.addImport("kv_cache", kv_cache_mod);
    exe.root_module.addImport("cudaz", cudaz_mod);
    exe.root_module.addImport("norm", norm_mod);
    exe.root_module.addImport("ffn", ffn_mod);
    exe.root_module.addImport("rope", rope_mod);
    exe.root_module.addImport("gqa", gqa_mod);
    exe.root_module.addImport("embedding", embedding_mod);
    exe.root_module.addImport("tokenizer", tokenizer_mod);
    exe.root_module.addImport("loader", loader_mod);
    exe.root_module.addImport("pipeline", pipeline_mod);

    if (ptx_output) |ptx| {
        exe.root_module.addAnonymousImport("flash_attention_ptx", .{ .root_source_file = ptx });
    }
    if (cubin_output) |cubin| {
        exe.root_module.addAnonymousImport("flash_attention_cubin", .{ .root_source_file = cubin });
    }
    if (dequant_ptx) |ptx| {
        exe.root_module.addAnonymousImport("dequantize_ptx", .{ .root_source_file = ptx });
    }

    if (has_cuda) {
        exe.linkSystemLibrary("cuda");
        exe.addLibraryPath(.{ .cwd_relative = cuda_path ++ "/lib64" });
        exe.addIncludePath(.{ .cwd_relative = cuda_path ++ "/include" });
        exe.linkSystemLibrary("cublas");
        exe.linkLibC();
    }

    b.installArtifact(exe);

    // === Run ===
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // === Tests ===
    const test_step = b.step("test", "Run all tests");

    const test_files = &.{
        "tests/test_tensor.zig",
        "tests/test_matmul.zig",
        "tests/test_flash_attention.zig",
        "tests/test_online_softmax.zig",
        "tests/test_transformer.zig",
        "tests/test_kv_cache.zig",
        "src/transformer/norm.zig",
        "src/transformer/ffn.zig",
        "src/transformer/rope.zig",
        "src/transformer/gqa.zig",
        "src/transformer/embedding.zig",
        "src/tokenizer/bpe.zig",
        "src/loader/safetensors.zig",
    };

    for (test_files) |tf| {
        const t = b.addTest(.{
            .root_source_file = b.path(tf),
            .target = target,
            .optimize = optimize,
        });
        t.root_module.addImport("core", core_mod);
        t.root_module.addImport("matmul", matmul_mod);
        t.root_module.addImport("fa", fa_mod);
        t.root_module.addImport("transformer", transformer_mod);
        t.root_module.addImport("kv_cache", kv_cache_mod);
        t.root_module.addImport("cudaz", cudaz_mod);
        t.root_module.addImport("norm", norm_mod);
        t.root_module.addImport("ffn", ffn_mod);
        t.root_module.addImport("rope", rope_mod);
        t.root_module.addImport("gqa", gqa_mod);
        t.root_module.addImport("embedding", embedding_mod);
        t.root_module.addImport("tokenizer", tokenizer_mod);
        t.root_module.addImport("loader", loader_mod);
        t.root_module.addImport("pipeline", pipeline_mod);
        if (has_cuda) {
            t.linkSystemLibrary("cuda");
            t.addLibraryPath(.{ .cwd_relative = cuda_path ++ "/lib64" });
            t.linkSystemLibrary("cublas");
            t.linkLibC();
        }
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }

    // === Benchmark ===
    const bench_step = b.step("bench", "Run benchmarks");
    const bench = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = b.path("tests/benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench.root_module.addImport("core", core_mod);
    bench.root_module.addImport("matmul", matmul_mod);
    bench.root_module.addImport("fa", fa_mod);
    bench.root_module.addImport("transformer", transformer_mod);
    bench.root_module.addImport("kv_cache", kv_cache_mod);
    bench.root_module.addImport("cudaz", cudaz_mod);
    if (has_cuda) {
        bench.linkSystemLibrary("cuda");
        bench.addLibraryPath(.{ .cwd_relative = cuda_path ++ "/lib64" });
        bench.linkSystemLibrary("cublas");
        bench.linkLibC();
    }
    b.installArtifact(bench);
    const run_bench = b.addRunArtifact(bench);
    bench_step.dependOn(&run_bench.step);
}
