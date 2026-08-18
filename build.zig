const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Detectar CUDA
    const cuda_path = b.option([]const u8, "cuda-path", "Path to CUDA installation")
        orelse b.graph.environ_map.get("CUDA_PATH") orelse "/usr/local/cuda";
    const cuda_bin_path = std.fmt.allocPrint(b.allocator, "{s}/bin", .{cuda_path}) catch "/usr/local/cuda/bin";
    const cuda_lib_path = std.fmt.allocPrint(b.allocator, "{s}/lib64", .{cuda_path}) catch "/usr/local/cuda/lib64";
    const cuda_inc_path = std.fmt.allocPrint(b.allocator, "{s}/include", .{cuda_path}) catch "/usr/local/cuda/include";

    const has_cuda = blk: {
        const nvcc = b.findProgram(&.{"nvcc"}, &.{cuda_bin_path}) catch break :blk false;
        _ = nvcc;
        break :blk true;
    };

    const io = b.graph.io;
    const cwd = std.Io.Dir.cwd();
    const cuda_lib_dir_exists = blk: {
        if (!has_cuda) break :blk false;
        std.Io.Dir.access(cwd, io, cuda_lib_path, .{}) catch break :blk false;
        break :blk true;
    };
    const cuda_inc_dir_exists = blk: {
        if (!has_cuda) break :blk false;
        std.Io.Dir.access(cwd, io, cuda_inc_path, .{}) catch break :blk false;
        break :blk true;
    };

    const has_openblas = b.option(bool, "openblas", "Link OpenBLAS") orelse false;

    // ── Arquitectura GPU ──────────────────────────────────────────────────────
    // Los cubins/PTX CUDA se compilan para la arquitectura detectada en el
    // host (compute capability vía `nvidia-smi`) en vez de hardcodear sm_86.
    // Se puede forzar con `-Dgpu-arch=sm_89` o la env `ZIG_AI_GPU_ARCH`.
    const gpu_arch = gpuArchDetect(b);
    const gpu_compute = gpuArchToCompute(b, gpu_arch);
    std.debug.print("GPU arch: {s} (kernels CUDA compilados para {s})\n", .{ gpu_arch, gpu_compute });

    var ptx_output: ?std.Build.LazyPath = null;
    var cubin_output: ?std.Build.LazyPath = null;
    var paged_cubin: ?std.Build.LazyPath = null;
    var layer_cubin: ?std.Build.LazyPath = null;
    // Objetos CUDA de dequantización (kernels/*.cu), compilados con nvcc y
    // enlazados al ejecutable/tests vía addObjectFile (patrón zig-cuda-agent).
    var dequant_objs: []const std.Build.LazyPath = &[0]std.Build.LazyPath{};

    if (has_cuda) {
        const nvcc_path = b.findProgram(&.{"nvcc"}, &.{cuda_bin_path}) catch unreachable;
        const arch_flag = std.fmt.allocPrint(b.allocator, "-arch={s}", .{gpu_compute}) catch unreachable;
        const code_flag = std.fmt.allocPrint(b.allocator, "-code={s}", .{gpu_arch}) catch unreachable;

        const compile_ptx = b.addSystemCommand(&.{
            nvcc_path,
            "-arch=compute_80", "-code=compute_80",
            "-ptx", "-o",
        });
        ptx_output = compile_ptx.addOutputFileArg("flash_attention.ptx");
        compile_ptx.addFileArg(b.path("cuda/flash_attention.cu"));

        const compile_cubin = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag, code_flag,
            "-cubin", "-o",
        });
        cubin_output = compile_cubin.addOutputFileArg("flash_attention.cubin");
        compile_cubin.addFileArg(b.path("cuda/flash_attention.cu"));

        // Cubin nativo para la GPU detectada: PagedAttention y layer_kernels.
        const compile_pa = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag, code_flag,
            "-cubin", "-o",
        });
        paged_cubin = compile_pa.addOutputFileArg("paged_attention.cubin");
        compile_pa.addFileArg(b.path("src/cuda/paged_attention.cu"));

        const compile_layer = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag, code_flag,
            "-cubin", "-o",
        });
        layer_cubin = compile_layer.addOutputFileArg("layer_kernels.cubin");
        compile_layer.addFileArg(b.path("src/cuda/layer_kernels.cu"));

        // Dequantización GGUF: compilar kernels/*.cu a objetos (patrón zig-cuda-agent).
        const dequant_sources = [_][]const u8{
            "dequant_q4_k.cu", "dequant_q6_k.cu", "dequant_iq4_xs.cu", "dequant_iq3_s.cu",
            "dequant_iq4_nl.cu", "dequant_iq2_xxs.cu", "dequant_iq2_xs.cu", "dequant_iq3_xxs.cu",
            "dequant_iq1_s.cu", "dequant_iq2_s.cu", "dequant_iq1_m.cu", "dequant_tq1_0.cu",
            "dequant_tq2_0.cu", "dequant_mxfp4.cu",
        };
        var objs: [dequant_sources.len]std.Build.LazyPath = undefined;
        inline for (dequant_sources, 0..) |src_name, i| {
            const nvcc = b.addSystemCommand(&.{
                nvcc_path, "-O3", "--use_fast_math",
                "-arch", gpu_arch, "-Xcompiler", "-fPIC", "-c", "-I", "kernels",
            });
            nvcc.addFileArg(b.path(b.pathJoin(&.{ "kernels", src_name })));
            nvcc.addArg("-o");
            objs[i] = nvcc.addOutputFileArg(src_name ++ ".o");
        }
        dequant_objs = &objs;
    }

    // === Módulo core ===
    const core_mod = b.createModule(.{
        .root_source_file = b.path("src/tensor.zig"),
        .target = target,
        .optimize = optimize,
    });

    // === Módulo gguf (debe definirse temprano: lo importan varios módulos) ===
    const gguf_mod = b.createModule(.{
        .root_source_file = b.path("src/loader/gguf.zig"),
        .target = target,
        .optimize = optimize,
    });

    // === Options: disponibilidad de CUDA para el módulo de dequant GPU ===
    const dequant_options = b.addOptions();
    dequant_options.addOption(bool, "has_cuda", has_cuda);

    // === Options: ruta al cubin de PagedAttention ===
    const paged_options = b.addOptions();
    var paged_cubin_install: ?*std.Build.Step.InstallFile = null;
    if (paged_cubin) |cb| {
        paged_cubin_install = b.addInstallFileWithDir(cb, .{ .custom = "lib" }, "paged_attention.cubin");
        b.getInstallStep().dependOn(&paged_cubin_install.?.step);
        paged_options.addOption([]const u8, "paged_cubin", b.getInstallPath(.{ .custom = "lib" }, "paged_attention.cubin"));
    } else {
        paged_options.addOption([]const u8, "paged_cubin", "");
    }

    // === Options: ruta al cubin de layer_kernels ===
    const layer_options = b.addOptions();
    if (layer_cubin) |cb| {
        const layer_install = b.addInstallFileWithDir(cb, .{ .custom = "lib" }, "layer_kernels.cubin");
        b.getInstallStep().dependOn(&layer_install.step);
        layer_options.addOption([]const u8, "layer_cubin", b.getInstallPath(.{ .custom = "lib" }, "layer_kernels.cubin"));
    } else {
        layer_options.addOption([]const u8, "layer_cubin", "");
    }

    // === Módulo time ===
    const time_mod = b.createModule(.{
        .root_source_file = b.path("src/utils/time.zig"),
        .target = target,
        .optimize = optimize,
    });

    // === Módulo matmul ===
    const options = b.addOptions();
    options.addOption(bool, "has_cuda", has_cuda);
    options.addOption(bool, "has_openblas", has_openblas);

    // === Módulo cudaz stub ===
    const cudaz_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/cudaz_stub.zig"),
    });

    // === Módulo CUDA Runtime API (patrón zig-cuda-agent) ===
    const cuda_runtime_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/cuda_runtime.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // === Módulo cublas (host) ===
    const cublas_mod = b.createModule(.{
        .root_source_file = b.path("src/matmul/cublas.zig"),
        .target = target,
        .optimize = optimize,
    });
    cublas_mod.addImport("core", core_mod);
    cublas_mod.addImport("cudaz", cudaz_mod);
    cublas_mod.addImport("time", time_mod);

    const matmul_mod = b.createModule(.{
        .root_source_file = b.path("src/matmul/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    matmul_mod.addImport("core", core_mod);
    matmul_mod.addImport("time", time_mod);
    matmul_mod.addImport("cudaz", cudaz_mod);
    matmul_mod.addImport("cublas", cublas_mod);
    matmul_mod.addOptions("build_options", options);

    // === Módulo dequant GPU de tensores GGUF ===
    const gguf_dequant_mod = b.createModule(.{
        .root_source_file = b.path("src/loader/gguf_dequant_gpu.zig"),
        .target = target,
        .optimize = optimize,
    });
    gguf_dequant_mod.addImport("cudaz", cudaz_mod);
    gguf_dequant_mod.addImport("cuda_runtime", cuda_runtime_mod);
    gguf_dequant_mod.addImport("gguf", gguf_mod);
    gguf_dequant_mod.addOptions("build_options", dequant_options);

    // === Módulo fa ===
    const fa_mod = b.createModule(.{
        .root_source_file = b.path("src/fa/flash_attention.zig"),
        .target = target,
        .optimize = optimize,
    });
    fa_mod.addImport("core", core_mod);
    fa_mod.addImport("matmul", matmul_mod);
    fa_mod.addImport("cudaz", cudaz_mod);
    fa_mod.addImport("time", time_mod);

    // === Módulo kv_cache ===
    const kv_cache_mod = b.createModule(.{
        .root_source_file = b.path("src/kv_cache.zig"),
        .target = target,
        .optimize = optimize,
    });
    kv_cache_mod.addImport("core", core_mod);
    kv_cache_mod.addImport("cudaz", cudaz_mod);

    // === Módulo norm ===
    const norm_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/norm.zig"),
        .target = target,
        .optimize = optimize,
    });
    norm_mod.addImport("core", core_mod);

    // === Módulo ffn ===
    const ffn_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/ffn.zig"),
        .target = target,
        .optimize = optimize,
    });
    ffn_mod.addImport("core", core_mod);
    ffn_mod.addImport("matmul", matmul_mod);

    // === Módulo embedding ===
    const embedding_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/embedding.zig"),
        .target = target,
        .optimize = optimize,
    });
    embedding_mod.addImport("core", core_mod);
    embedding_mod.addImport("matmul", matmul_mod);

    // === Módulo rope ===
    const rope_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/rope.zig"),
        .target = target,
        .optimize = optimize,
    });
    rope_mod.addImport("core", core_mod);

    // === Módulo gqa ===
    const gqa_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/gqa.zig"),
        .target = target,
        .optimize = optimize,
    });
    gqa_mod.addImport("core", core_mod);

    // === Módulo loader ===
    const loader_mod = b.createModule(.{
        .root_source_file = b.path("src/loader/safetensors.zig"),
        .target = target,
        .optimize = optimize,
    });
    loader_mod.addImport("core", core_mod);

    // === Módulo model_config ===
    const model_config_mod = b.createModule(.{
        .root_source_file = b.path("src/loader/model_config.zig"),
        .target = target,
        .optimize = optimize,
    });
    model_config_mod.addImport("gguf", gguf_mod);

    // === Módulo quant_weight ===
    const quant_weight_mod = b.createModule(.{
        .root_source_file = b.path("src/loader/quant_weight.zig"),
        .target = target,
        .optimize = optimize,
    });
    quant_weight_mod.addImport("gguf", gguf_mod);

    // === Módulo gguf_model ===
    const gguf_model_mod = b.createModule(.{
        .root_source_file = b.path("src/loader/gguf_model.zig"),
        .target = target,
        .optimize = optimize,
    });
    gguf_model_mod.addImport("gguf", gguf_mod);
    gguf_model_mod.addImport("model_config", model_config_mod);
    gguf_model_mod.addImport("core", core_mod);
    gguf_model_mod.addImport("quant_weight", quant_weight_mod);

    // === Módulo debug (breadcrumbs de diagnóstico centralizados) ===
    const debug_mod = b.createModule(.{
        .root_source_file = b.path("src/debug.zig"),
        .target = target,
        .optimize = optimize,
    });

    // === Módulo layer_kernels (elementwise GPU para capa híbrida residente) ===
    const layer_kernels_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/layer_kernels.zig"),
        .target = target,
        .optimize = optimize,
    });
    layer_kernels_mod.addImport("cudaz", cudaz_mod);
    layer_kernels_mod.addOptions("build_options", layer_options);
    layer_kernels_mod.addImport("debug", debug_mod);

    // === Módulo decode_graph (CUDA Graphs para el decode por token) ===
    const decode_graph_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/decode_graph.zig"),
        .target = target,
        .optimize = optimize,
    });
    decode_graph_mod.addImport("cudaz", cudaz_mod);
    decode_graph_mod.addImport("debug", debug_mod);
    decode_graph_mod.addImport("layer_kernels", layer_kernels_mod);

    // === Módulo ssm ===
    const ssm_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/ssm.zig"),
        .target = target,
        .optimize = optimize,
    });
    ssm_mod.addImport("core", core_mod);
    ssm_mod.addImport("matmul", matmul_mod);
    ssm_mod.addImport("cublas", cublas_mod);
    ssm_mod.addImport("cudaz", cudaz_mod);
    ssm_mod.addImport("layer_kernels", layer_kernels_mod);
    ssm_mod.addImport("quant_weight", quant_weight_mod);
    ssm_mod.addImport("gguf", gguf_mod);
    ssm_mod.addImport("debug", debug_mod);

    // === Módulo hybrid_attn ===
    const hybrid_attn_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/hybrid_attn.zig"),
        .target = target,
        .optimize = optimize,
    });
    hybrid_attn_mod.addImport("core", core_mod);
    hybrid_attn_mod.addImport("matmul", matmul_mod);
    hybrid_attn_mod.addImport("quant_weight", quant_weight_mod);
    hybrid_attn_mod.addImport("gguf", gguf_mod);
    hybrid_attn_mod.addImport("norm", norm_mod);
    hybrid_attn_mod.addImport("ffn", ffn_mod);
    hybrid_attn_mod.addImport("rope", rope_mod);
    hybrid_attn_mod.addImport("kv_cache", kv_cache_mod);
    hybrid_attn_mod.addImport("cublas", cublas_mod);
    hybrid_attn_mod.addImport("cudaz", cudaz_mod);
    hybrid_attn_mod.addImport("layer_kernels", layer_kernels_mod);
    hybrid_attn_mod.addImport("debug", debug_mod);

    // === Módulo hybrid_layer ===
    const hybrid_layer_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/hybrid_layer.zig"),
        .target = target,
        .optimize = optimize,
    });
    hybrid_layer_mod.addImport("core", core_mod);
    hybrid_layer_mod.addImport("matmul", matmul_mod);
    hybrid_layer_mod.addImport("quant_weight", quant_weight_mod);
    hybrid_layer_mod.addImport("gguf", gguf_mod);
    hybrid_layer_mod.addImport("norm", norm_mod);
    hybrid_layer_mod.addImport("ffn", ffn_mod);
    hybrid_layer_mod.addImport("rope", rope_mod);
    hybrid_layer_mod.addImport("model_config", model_config_mod);
    hybrid_layer_mod.addImport("hybrid_attn", hybrid_attn_mod);
    hybrid_layer_mod.addImport("ssm", ssm_mod);
    hybrid_layer_mod.addImport("cublas", cublas_mod);
    hybrid_layer_mod.addImport("cudaz", cudaz_mod);
    hybrid_layer_mod.addImport("layer_kernels", layer_kernels_mod);
    hybrid_layer_mod.addImport("debug", debug_mod);

    // === Módulo gguf_tokenizer ===
    const gguf_tokenizer_mod = b.createModule(.{
        .root_source_file = b.path("src/loader/gguf_tokenizer.zig"),
        .target = target,
        .optimize = optimize,
    });
    gguf_tokenizer_mod.addImport("gguf", gguf_mod);

    // === Módulo unicode_data (tablas generadas) ===
    const unicode_data_mod = b.createModule(.{
        .root_source_file = b.path("src/tokenizer/unicode_data.zig"),
        .target = target,
        .optimize = optimize,
    });

    // === Módulo unicode (clasificación/pre-tokenización) ===
    const unicode_mod = b.createModule(.{
        .root_source_file = b.path("src/tokenizer/unicode.zig"),
        .target = target,
        .optimize = optimize,
    });
    unicode_mod.addImport("unicode_data", unicode_data_mod);

    // === Módulo tokenizer ===
    const tokenizer_mod = b.createModule(.{
        .root_source_file = b.path("src/tokenizer/bpe.zig"),
        .target = target,
        .optimize = optimize,
    });
    tokenizer_mod.addImport("gguf_tokenizer", gguf_tokenizer_mod);
    tokenizer_mod.addImport("unicode", unicode_mod);

    // === Módulo paged_attention ===
    const paged_attention_mod = b.createModule(.{
        .root_source_file = b.path("src/paged_attention/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    paged_attention_mod.addImport("cudaz", cudaz_mod);
    paged_attention_mod.addImport("kv_cache", kv_cache_mod);
    paged_attention_mod.addOptions("build_options", paged_options);

    // hybrid_attn usa PagedKVCache para el KV-cache de atención
    hybrid_attn_mod.addImport("paged_attention", paged_attention_mod);
    hybrid_layer_mod.addImport("paged_attention", paged_attention_mod);

    // === Módulo transformer ===
    const transformer_mod = b.createModule(.{
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
    transformer_mod.addImport("cudaz", cudaz_mod);
    transformer_mod.addImport("hybrid_layer", hybrid_layer_mod);
    transformer_mod.addImport("gguf", gguf_mod);

    // === Módulo pipeline ===
    const pipeline_mod = b.createModule(.{
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
    pipeline_mod.addImport("time", time_mod);
    pipeline_mod.addImport("gguf", gguf_mod);
    pipeline_mod.addImport("model_config", model_config_mod);

    // === Ejecutable principal ===
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe_mod.addImport("core", core_mod);
    exe_mod.addImport("matmul", matmul_mod);
    exe_mod.addImport("fa", fa_mod);
    exe_mod.addImport("transformer", transformer_mod);
    exe_mod.addImport("kv_cache", kv_cache_mod);
    exe_mod.addImport("cudaz", cudaz_mod);
    exe_mod.addImport("norm", norm_mod);
    exe_mod.addImport("ffn", ffn_mod);
    exe_mod.addImport("rope", rope_mod);
    exe_mod.addImport("gqa", gqa_mod);
    exe_mod.addImport("embedding", embedding_mod);
    exe_mod.addImport("tokenizer", tokenizer_mod);
    exe_mod.addImport("loader", loader_mod);
    exe_mod.addImport("gguf", gguf_mod);
    exe_mod.addImport("model_config", model_config_mod);
    exe_mod.addImport("gguf_model", gguf_model_mod);
    exe_mod.addImport("gguf_tokenizer", gguf_tokenizer_mod);
    exe_mod.addImport("pipeline", pipeline_mod);
    exe_mod.addImport("paged_attention", paged_attention_mod);
    exe_mod.addImport("layer_kernels", layer_kernels_mod);
    exe_mod.addImport("decode_graph", decode_graph_mod);
    exe_mod.addImport("cublas", cublas_mod);
    exe_mod.addImport("time", time_mod);
    exe_mod.addImport("debug", debug_mod);

    if (ptx_output) |ptx| {
        exe_mod.addAnonymousImport("flash_attention_ptx", .{ .root_source_file = ptx });
    }
    if (cubin_output) |cubin| {
        exe_mod.addAnonymousImport("flash_attention_cubin", .{ .root_source_file = cubin });
    }

    exe_mod.link_libc = true;
    if (has_cuda) {
        exe_mod.linkSystemLibrary("cuda", .{});
        exe_mod.linkSystemLibrary("cudart", .{});
        if (cuda_lib_dir_exists) exe_mod.addLibraryPath(.{ .cwd_relative = cuda_lib_path });
        if (cuda_inc_dir_exists) exe_mod.addIncludePath(.{ .cwd_relative = cuda_inc_path });
        exe_mod.linkSystemLibrary("cublas", .{});
        for (dequant_objs) |obj| {
            exe_mod.addObjectFile(obj);
        }
    }

    const exe = b.addExecutable(.{
        .name = "zig-ai-engine",
        .root_module = exe_mod,
    });

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
        "tests/test_paged_attention.zig",
        "tests/test_gguf.zig",
        "tests/test_paged_attention_gpu.zig",
        "src/transformer/norm.zig",
        "src/transformer/ffn.zig",
        "src/transformer/rope.zig",
        "src/transformer/gqa.zig",
        "src/transformer/embedding.zig",
        "src/transformer/ssm.zig",
        "src/transformer/hybrid_attn.zig",
        "src/transformer/hybrid_layer.zig",
        "src/tokenizer/bpe.zig",
        "src/loader/safetensors.zig",
        "src/loader/gguf.zig",
        "src/loader/quant_weight.zig",
        "src/loader/model_config.zig",
        "src/loader/gguf_tokenizer.zig",
        "src/loader/gguf_model.zig",
        "tests/test_dequant_gpu.zig",
    };

    inline for (test_files) |tf| {
        const tmod = b.createModule(.{
            .root_source_file = b.path(tf),
            .target = target,
            .optimize = optimize,
        });
        tmod.addImport("core", core_mod);
        tmod.addImport("matmul", matmul_mod);
        tmod.addImport("fa", fa_mod);
        tmod.addImport("transformer", transformer_mod);
        tmod.addImport("kv_cache", kv_cache_mod);
        tmod.addImport("cudaz", cudaz_mod);
        tmod.addImport("norm", norm_mod);
        tmod.addImport("ffn", ffn_mod);
        tmod.addImport("rope", rope_mod);
        tmod.addImport("gqa", gqa_mod);
        tmod.addImport("embedding", embedding_mod);
        tmod.addImport("hybrid_attn", hybrid_attn_mod);
        tmod.addImport("hybrid_layer", hybrid_layer_mod);
        tmod.addImport("tokenizer", tokenizer_mod);
        tmod.addImport("gguf", gguf_mod);
        tmod.addImport("quant_weight", quant_weight_mod);
        tmod.addImport("model_config", model_config_mod);
        tmod.addImport("gguf_tokenizer", gguf_tokenizer_mod);
        tmod.addImport("unicode", unicode_mod);
        tmod.addImport("unicode_data", unicode_data_mod);
        tmod.addImport("gguf_model", gguf_model_mod);
        tmod.addImport("gguf_dequant", gguf_dequant_mod);
        tmod.addImport("pipeline", pipeline_mod);
        tmod.addImport("paged_attention", paged_attention_mod);
        tmod.addImport("time", time_mod);
        tmod.addImport("cublas", cublas_mod);
        tmod.addImport("layer_kernels", layer_kernels_mod);

        const t = b.addTest(.{ .root_module = tmod });
        tmod.link_libc = true;
        if (has_cuda) {
            tmod.linkSystemLibrary("cuda", .{});
            tmod.linkSystemLibrary("cudart", .{});
            if (cuda_lib_dir_exists) tmod.addLibraryPath(.{ .cwd_relative = cuda_lib_path });
            tmod.linkSystemLibrary("cublas", .{});
            for (dequant_objs) |obj| {
                tmod.addObjectFile(obj);
            }
        }
        const run_t = b.addRunArtifact(t);
        if (paged_cubin_install) |inst| {
            run_t.step.dependOn(&inst.step);
        }
        test_step.dependOn(&run_t.step);
    }

    // === Benchmark ===
    const bench_step = b.step("bench", "Run benchmarks");
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("tests/benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_mod.addImport("core", core_mod);
    bench_mod.addImport("matmul", matmul_mod);
    bench_mod.addImport("fa", fa_mod);
    bench_mod.addImport("transformer", transformer_mod);
    bench_mod.addImport("kv_cache", kv_cache_mod);
    bench_mod.addImport("cudaz", cudaz_mod);
    bench_mod.addImport("time", time_mod);
    bench_mod.link_libc = true;
    if (has_cuda) {
        bench_mod.linkSystemLibrary("cuda", .{});
        bench_mod.linkSystemLibrary("cudart", .{});
        if (cuda_lib_dir_exists) bench_mod.addLibraryPath(.{ .cwd_relative = cuda_lib_path });
        bench_mod.linkSystemLibrary("cublas", .{});
    }
    const bench = b.addExecutable(.{
        .name = "benchmark",
        .root_module = bench_mod,
    });
    b.installArtifact(bench);
    const run_bench = b.addRunArtifact(bench);
    bench_step.dependOn(&run_bench.step);

    // === Benchmark PagedAttention ===
    const bench_pa_step = b.step("bench-pa", "Run PagedAttention benchmarks");
    const bench_pa_mod = b.createModule(.{
        .root_source_file = b.path("benchmarks/bench_paged_attention.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_pa_mod.addImport("paged_attention", paged_attention_mod);
    bench_pa_mod.addImport("cudaz", cudaz_mod);
    bench_pa_mod.addImport("time", time_mod);
    bench_pa_mod.link_libc = true;
    if (has_cuda) {
        bench_pa_mod.linkSystemLibrary("cuda", .{});
        bench_pa_mod.linkSystemLibrary("cudart", .{});
        if (cuda_lib_dir_exists) bench_pa_mod.addLibraryPath(.{ .cwd_relative = cuda_lib_path });
        bench_pa_mod.linkSystemLibrary("cublas", .{});
    }
    const bench_pa = b.addExecutable(.{
        .name = "bench_paged_attention",
        .root_module = bench_pa_mod,
    });
    b.installArtifact(bench_pa);
    const run_bench_pa = b.addRunArtifact(bench_pa);
    if (paged_cubin_install) |inst| {
        run_bench_pa.step.dependOn(&inst.step);
    }
    bench_pa_step.dependOn(&run_bench_pa.step);
}

/// Detección de la arquitectura GPU (p.ej. "sm_86") para compilar los kernels
/// CUDA. Orden de prioridad:
///   1. `-Dgpu-arch=sm_XX` (opción de build)
///   2. env `ZIG_AI_GPU_ARCH`
///   3. auto-detección vía `nvidia-smi --query-gpu=compute_cap`
///   4. fallback "sm_86" (con aviso; forzar vía `-Dgpu-arch` si es otra GPU).
fn gpuArchDetect(b: *std.Build) []const u8 {
    if (b.option([]const u8, "gpu-arch", "Arquitectura GPU para compilar kernels CUDA (p.ej. sm_86). Se auto-detecta vía nvidia-smi si no se indica.")) |a|
        return a;
    if (b.graph.environ_map.get("ZIG_AI_GPU_ARCH")) |a|
        return a;
    if (!std.process.can_spawn) return "sm_86";
    var code: u8 = undefined;
    const out = b.runAllowFail(&.{ "nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader" }, &code, .ignore) catch return "sm_86";
    defer b.allocator.free(out);
    var it0 = std.mem.splitScalar(u8, out, '\n');
    const line0 = it0.next() orelse return "sm_86";
    const cap = std.mem.trim(u8, line0, " \t\r");
    if (cap.len < 3) return "sm_86";
    var it = std.mem.splitScalar(u8, cap, '.');
    const maj = it.next() orelse return "sm_86";
    const min = it.next() orelse return "sm_86";
    return std.fmt.allocPrint(b.allocator, "sm_{s}{s}", .{ maj, std.mem.trim(u8, min, " \t\r") }) catch "sm_86";
}

/// "sm_86" -> "compute_86"
fn gpuArchToCompute(b: *std.Build, arch: []const u8) []const u8 {
    if (arch.len > 3 and std.mem.startsWith(u8, arch, "sm_"))
        return std.fmt.allocPrint(b.allocator, "compute_{s}", .{arch[3..]}) catch arch;
    return arch;
}
