const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Post-freeze 2026-09-13 (coordinador): `--test-filter <tag>` documentado en
    // AGENTS.md/TODOs PERO inexistente como flag del build runner 0.16
    // ("unrecognized argument" — es flag del DRIVER `zig test`). API real:
    // Compile.filters (std/Build/Step/Compile.zig:59). Lo exponemos como
    // option de build para que el protocolo CPU de lanes funcione tal como
    // está documentado: `zig build test --test-filter kv_quant` (repeatible
    // para OR de tags). Aplica a TODOS los test steps (principal + dbg).
    const test_filters: []const []const u8 = blk: {
        const opt = b.option([]const u8, "test-filter", "Solo tests cuyo nombre matchee (substring, OR si se repite) — protocolo CPU lanes") orelse break :blk &.{};
        const one = b.allocator.alloc([]const u8, 1) catch break :blk &.{};
        one[0] = opt;
        break :blk one;
    };

    // Detectar CUDA
    const cuda_path = b.option([]const u8, "cuda-path", "Path to CUDA installation") orelse b.graph.environ_map.get("CUDA_PATH") orelse "/usr/local/cuda";
    const cuda_lib_path = std.fmt.allocPrint(b.allocator, "{s}/lib64", .{cuda_path}) catch "/usr/local/cuda/lib64";
    const cuda_inc_path = std.fmt.allocPrint(b.allocator, "{s}/include", .{cuda_path}) catch "/usr/local/cuda/include";

    const has_cuda = blk: {
        const nvcc = b.findProgram(&[_][]const u8{"nvcc"}, &[_][]const u8{}) catch break :blk false;
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
    var extra_cubin: ?std.Build.LazyPath = null;
    var moe_cubin: ?std.Build.LazyPath = null; // lane-e E3
    var fp8_cubin: ?std.Build.LazyPath = null; // lane-b FP8 block
    var prefill_cubin: ?std.Build.LazyPath = null; // STUDY §5.2 (Dev C)
    var kvarn_cubin: ?std.Build.LazyPath = null; // lane-b1 (Dev A) KVarN store/materialize
    var kvarn_mma_cubin: ?std.Build.LazyPath = null; // lane-b1 (Dev A) A9 MMA smoke
    var fattn_kvarn_portable_cubin: ?std.Build.LazyPath = null; // lane-b1 (Dev B) portable FA
    var fattn_kvarn_vec_cubin: ?std.Build.LazyPath = null; // lane-b1 (Dev B) vec FA D256 SWA
    // Objetos CUDA de dequantización (kernels/*.cu), compilados con nvcc y
    // enlazados al ejecutable/tests vía addObjectFile (patrón zig-cuda-agent).
    var dequant_objs: []const std.Build.LazyPath = &[0]std.Build.LazyPath{};
    var hybrid_split_obj: ?std.Build.LazyPath = null;
    var quant_encode_obj: ?std.Build.LazyPath = null; // lane-f Phase 3
    var kvarn_split_cubin: ?std.Build.LazyPath = null; // lane-b1 (Dev A) A10

    if (has_cuda) {
        const nvcc_path = b.findProgram(&[_][]const u8{"nvcc"}, &[_][]const u8{}) catch unreachable;
        const arch_flag = std.fmt.allocPrint(b.allocator, "-arch={s}", .{gpu_compute}) catch unreachable;
        const code_flag = std.fmt.allocPrint(b.allocator, "-code={s}", .{gpu_arch}) catch unreachable;

        const compile_ptx = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag,
            "-ptx",
            "-o",
        });
        ptx_output = compile_ptx.addOutputFileArg("flash_attention.ptx");
        compile_ptx.addFileArg(b.path("cuda/flash_attention.cu"));

        const compile_cubin = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag,
            code_flag,
            "-cubin",
            "-o",
        });
        cubin_output = compile_cubin.addOutputFileArg("flash_attention.cubin");
        compile_cubin.addFileArg(b.path("cuda/flash_attention.cu"));

        // Cubin nativo para la GPU detectada: PagedAttention y layer_kernels.
        const compile_pa = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag,
            code_flag,
            "-cubin",
            "-o",
        });
        paged_cubin = compile_pa.addOutputFileArg("paged_attention.cubin");
        compile_pa.addFileArg(b.path("src/cuda/paged_attention.cu"));

        // Lane-b1 (Dev A): kernels KVarN store/materialize sobre layout C1.
        const compile_kvarn = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag,
            code_flag,
            "-cubin",
            "-o",
        });
        kvarn_cubin = compile_kvarn.addOutputFileArg("kvarn_kernels.cubin");
        compile_kvarn.addFileArg(b.path("src/cuda/kvarn_kernels.cu"));

        // Lane-b1 (Dev A) A9: smoke del módulo MMA (m16n8k16).
        const compile_mma = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag,
            code_flag,
            "-cubin",
            "-I",
            "src/cuda",
            "-o",
        });
        kvarn_mma_cubin = compile_mma.addOutputFileArg("kvarn_mma_smoke.cubin");
        compile_mma.addFileArg(b.path("src/cuda/kvarn_mma_smoke.cu"));

        // Lane-b1 (Dev B): FA portable sobre records KVarN (ruta universal).
        const compile_fattn_portable = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag,
            code_flag,
            "-cubin",
            "-I",
            "src/cuda",
            "-o",
        });
        fattn_kvarn_portable_cubin = compile_fattn_portable.addOutputFileArg("fattn_kvarn_portable.cubin");
        compile_fattn_portable.addFileArg(b.path("src/cuda/fattn_kvarn_portable.cu"));

        // Lane-b1 (Dev B): FA vec D256 SWA (GQA2, splits 8/16/32).
        const compile_fattn_vec = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag,
            code_flag,
            "-cubin",
            "-I",
            "src/cuda",
            "-o",
        });
        fattn_kvarn_vec_cubin = compile_fattn_vec.addOutputFileArg("fattn_kvarn_vec.cubin");
        compile_fattn_vec.addFileArg(b.path("src/cuda/fattn_kvarn_vec.cu"));

        // Lane A: formatos extra (iq4_xs, q4_1, q5_0/51, q8_1) en cubin propio.
        const compile_extra = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag,
            code_flag,
            "-cubin",
            "-I",
            "kernels",
            "-o",
        });
        extra_cubin = compile_extra.addOutputFileArg("fused_decode_extra.cubin");
        compile_extra.addFileArg(b.path("src/cuda/fused_decode_extra.cu"));

        const compile_layer = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag,
            code_flag,
            "-cubin",
            "-o",
        });
        layer_cubin = compile_layer.addOutputFileArg("layer_kernels.cubin");
        compile_layer.addFileArg(b.path("src/cuda/layer_kernels.cu"));

        // STUDY §5.2: chunked batched ΔNet prefill (Dev C) — cubin propio
        // (cubin único con dos -o no soportado: source separada, artefacto
        // separado, dispatch por ssmlayer.prefill).
        const compile_prefill = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag,
            code_flag,
            "-cubin",
            "-o",
        });
        prefill_cubin = compile_prefill.addOutputFileArg("prefill_delta_net.cubin");
        compile_prefill.addFileArg(b.path("src/cuda/prefill_delta_net.cu"));

        // Lane E (E3): kernels MoE cubin propio.
        const compile_moe = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag,
            code_flag,
            "-cubin",
            "-o",
        });
        moe_cubin = compile_moe.addOutputFileArg("moe_kernels.cubin");
        compile_moe.addFileArg(b.path("src/cuda/moe_kernels.cu"));

        // Compile hybrid_split.cu to an object file for the hybrid kernel launcher
        const compile_hybrid_split = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag,
            code_flag,
            "-c",
            "-o",
        });
        hybrid_split_obj = compile_hybrid_split.addOutputFileArg("hybrid_split.cu.o");
        compile_hybrid_split.addFileArg(b.path("src/moe/hybrid_split.cu"));

        // lane-f Phase 3: kernels de encode (MXFP4/Q8_0) a objeto (Runtime API).
        const compile_quant_encode = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag,
            code_flag,
            "-O3",
            "-c",
            "-o",
        });
        quant_encode_obj = compile_quant_encode.addOutputFileArg("encode_kernels.cu.o");
        compile_quant_encode.addFileArg(b.path("src/matmul/quant/encode_kernels.cu"));

        // Lane-b1 (Dev A) A10: decode-split MMA (cubin Driver API; los
        // templates se instancian tras wrappers __global__ extern "C").
        const compile_kvarn_split = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag,
            code_flag,
            "-O3",
            "-cubin",
            "-I",
            "src/cuda",
            "-o",
        });
        kvarn_split_cubin = compile_kvarn_split.addOutputFileArg("fattn_kvarn_split.cubin");
        compile_kvarn_split.addFileArg(b.path("src/cuda/fattn_kvarn_split.cu"));

        // FP8 Block-Scaled Linear kernels (Lane B)
        const compile_fp8 = b.addSystemCommand(&.{
            nvcc_path,
            arch_flag,
            code_flag,
            "-cubin",
            "--use_fast_math",
            "-o",
        });
        fp8_cubin = compile_fp8.addOutputFileArg("fp8_block_kernels.cubin");
        compile_fp8.addFileArg(b.path("src/cuda/fp8_block_kernels.cu"));

        // Dequantización GGUF: compilar kernels/*.cu a objetos (patrón zig-cuda-agent).
        const dequant_sources = [_][]const u8{
            "dequant_q4_k.cu",     "dequant_q6_k.cu",      "dequant_iq4_xs.cu", "dequant_iq3_s.cu",
            "dequant_iq4_nl.cu",   "dequant_iq2_xxs.cu",   "dequant_iq2_xs.cu", "dequant_iq3_xxs.cu",
            "dequant_iq1_s.cu",    "dequant_iq2_s.cu",     "dequant_iq1_m.cu",  "dequant_tq1_0.cu",
            "dequant_tq2_0.cu",    "dequant_mxfp4.cu",     "dequant_q4_0.cu",   "dequant_q4_1.cu",
            "dequant_q5_0.cu",     "dequant_q5_1.cu",      "dequant_q8_0.cu",   "dequant_q8_1.cu",
            "dequant_q2_k.cu",     "dequant_q3_k.cu",      "dequant_q5_k.cu",   "dequant_q8_k.cu",
            "dequant_int8_sym.cu", "dequant_int8_asym.cu", "dequant_int4.cu",
        };
        var objs: [dequant_sources.len]std.Build.LazyPath = undefined;
        inline for (dequant_sources, 0..) |src_name, i| {
            const nvcc = b.addSystemCommand(&.{
                nvcc_path, "-O3",    "--use_fast_math",
                "-arch",   gpu_arch, "-Xcompiler",
                "-fPIC",   "-c",     "-I",
                "kernels",
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
    if (extra_cubin) |ec| {
        const inst = b.addInstallFileWithDir(ec, .{ .custom = "lib" }, "fused_decode_extra.cubin");
        b.getInstallStep().dependOn(&inst.step);
        paged_options.addOption([]const u8, "fused_extra_cubin", b.getInstallPath(.{ .custom = "lib" }, "fused_decode_extra.cubin"));
    } else {
        paged_options.addOption([]const u8, "fused_extra_cubin", "");
    }

    // === Options: ruta al cubin KVarN (lane-b1 Dev A) ===
    const kvarn_options = b.addOptions();
    var kvarn_cubin_install: ?*std.Build.Step.InstallFile = null;
    if (kvarn_cubin) |cb| {
        kvarn_cubin_install = b.addInstallFileWithDir(cb, .{ .custom = "lib" }, "kvarn_kernels.cubin");
        b.getInstallStep().dependOn(&kvarn_cubin_install.?.step);
        kvarn_options.addOption([]const u8, "kvarn_cubin", b.getInstallPath(.{ .custom = "lib" }, "kvarn_kernels.cubin"));
    } else {
        kvarn_options.addOption([]const u8, "kvarn_cubin", "");
    }
    // === Options: cubin decode-split (lane-b1 Dev A, A10) ===
    var kvarn_split_cubin_install: ?*std.Build.Step.InstallFile = null;
    if (kvarn_split_cubin) |cb| {
        kvarn_split_cubin_install = b.addInstallFileWithDir(cb, .{ .custom = "lib" }, "fattn_kvarn_split.cubin");
        b.getInstallStep().dependOn(&kvarn_split_cubin_install.?.step);
        kvarn_options.addOption([]const u8, "kvarn_split_cubin", b.getInstallPath(.{ .custom = "lib" }, "fattn_kvarn_split.cubin"));
    } else {
        kvarn_options.addOption([]const u8, "kvarn_split_cubin", "");
    }
    // === Options: cubin MMA smoke (lane-b1 Dev A, A9) ===
    var kvarn_mma_cubin_install: ?*std.Build.Step.InstallFile = null;
    if (kvarn_mma_cubin) |cb| {
        kvarn_mma_cubin_install = b.addInstallFileWithDir(cb, .{ .custom = "lib" }, "kvarn_mma_smoke.cubin");
        b.getInstallStep().dependOn(&kvarn_mma_cubin_install.?.step);
        kvarn_options.addOption([]const u8, "kvarn_mma_cubin", b.getInstallPath(.{ .custom = "lib" }, "kvarn_mma_smoke.cubin"));
    } else {
        kvarn_options.addOption([]const u8, "kvarn_mma_cubin", "");
    }
    // === Options: rutas a los cubins FA KVarN (lane-b1 Dev-B) ===
    var fattn_kvarn_portable_install: ?*std.Build.Step.InstallFile = null;
    var fattn_kvarn_vec_install: ?*std.Build.Step.InstallFile = null;
    if (fattn_kvarn_portable_cubin) |cb| {
        fattn_kvarn_portable_install = b.addInstallFileWithDir(cb, .{ .custom = "lib" }, "fattn_kvarn_portable.cubin");
        b.getInstallStep().dependOn(&fattn_kvarn_portable_install.?.step);
        kvarn_options.addOption([]const u8, "fattn_kvarn_portable_cubin", b.getInstallPath(.{ .custom = "lib" }, "fattn_kvarn_portable.cubin"));
    } else {
        kvarn_options.addOption([]const u8, "fattn_kvarn_portable_cubin", "");
    }
    if (fattn_kvarn_vec_cubin) |cb| {
        fattn_kvarn_vec_install = b.addInstallFileWithDir(cb, .{ .custom = "lib" }, "fattn_kvarn_vec.cubin");
        b.getInstallStep().dependOn(&fattn_kvarn_vec_install.?.step);
        kvarn_options.addOption([]const u8, "fattn_kvarn_vec_cubin", b.getInstallPath(.{ .custom = "lib" }, "fattn_kvarn_vec.cubin"));
    } else {
        kvarn_options.addOption([]const u8, "fattn_kvarn_vec_cubin", "");
    }
    // Alias con el nombre que asumen los tests de Dev-B (portable cubin
    // contiene wht + portable kernels; el vec usa su propio option).
    if (fattn_kvarn_portable_cubin != null) {
        kvarn_options.addOption([]const u8, "fattn_cubin", b.getInstallPath(.{ .custom = "lib" }, "fattn_kvarn_portable.cubin"));
    } else {
        kvarn_options.addOption([]const u8, "fattn_cubin", "");
    }

    // === Options: ruta al cubin de layer_kernels ===
    const layer_options = b.addOptions();
    var layer_cubin_install: ?*std.Build.Step.InstallFile = null;
    if (layer_cubin) |cb| {
        layer_cubin_install = b.addInstallFileWithDir(cb, .{ .custom = "lib" }, "layer_kernels.cubin");
        b.getInstallStep().dependOn(&layer_cubin_install.?.step);
        layer_options.addOption([]const u8, "layer_cubin", b.getInstallPath(.{ .custom = "lib" }, "layer_kernels.cubin"));
    } else {
        layer_options.addOption([]const u8, "layer_cubin", "");
    }
    // STUDY §5.2: cubin propio del prefill chunked (Dev C).
    var prefill_cubin_install: ?*std.Build.Step.InstallFile = null;
    if (prefill_cubin) |cb| {
        prefill_cubin_install = b.addInstallFileWithDir(cb, .{ .custom = "lib" }, "prefill_delta_net.cubin");
        b.getInstallStep().dependOn(&prefill_cubin_install.?.step);
        layer_options.addOption([]const u8, "prefill_cubin", b.getInstallPath(.{ .custom = "lib" }, "prefill_delta_net.cubin"));
    } else {
        layer_options.addOption([]const u8, "prefill_cubin", "");
    }

    // === Options: ruta al cubin de moe_kernels (lane-e E3) ===
    const moe_options = b.addOptions();
    var moe_cubin_install: ?*std.Build.Step.InstallFile = null;
    if (moe_cubin) |cb| {
        moe_cubin_install = b.addInstallFileWithDir(cb, .{ .custom = "lib" }, "moe_kernels.cubin");
        b.getInstallStep().dependOn(&moe_cubin_install.?.step);
        moe_options.addOption([]const u8, "moe_cubin", b.getInstallPath(.{ .custom = "lib" }, "moe_kernels.cubin"));
    } else {
        moe_options.addOption([]const u8, "moe_cubin", "");
    }

    // === Options: ruta al cubin de FP8 block kernels (lane-b) ===
    const fp8_options = b.addOptions();
    var fp8_cubin_install: ?*std.Build.Step.InstallFile = null;
    if (fp8_cubin) |cb| {
        fp8_cubin_install = b.addInstallFileWithDir(cb, .{ .custom = "lib" }, "fp8_block_kernels.cubin");
        b.getInstallStep().dependOn(&fp8_cubin_install.?.step);
        fp8_options.addOption([]const u8, "fp8_cubin", b.getInstallPath(.{ .custom = "lib" }, "fp8_block_kernels.cubin"));
    } else {
        fp8_options.addOption([]const u8, "fp8_cubin", "");
    }

    // === Módulo time ===
    const time_mod = b.createModule(.{
        .root_source_file = b.path("src/utils/time.zig"),
        .target = target,
        .optimize = optimize,
    });

    // === Módulo resources (coordinador, post-freeze 2026-09-13) ===
    // Presupuesto CPU/mem central: colapso SMT, loadavg, MemAvailable,
    // ZIG_AI_CPU_WORKERS global + breadcrumb [cpu_budget]. Consumidores:
    // matmul, vision, moe/cpu_executor, inference.
    const resources_mod = b.createModule(.{
        .root_source_file = b.path("src/utils/resources.zig"),
        .target = target,
        .optimize = optimize,
    });
    resources_mod.link_libc = true; // extern sched_getaffinity/getenv

    // === Módulo matmul ===
    const options = b.addOptions();
    options.addOption(bool, "has_cuda", has_cuda);
    options.addOption(bool, "has_openblas", has_openblas);

    // === Módulo cudaz stub ===
    const cudaz_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/cudaz_stub.zig"),
    });

    // === Módulo NVRTC JIT (lane-cuda UC-1) ===
    // Runtime-compile de kernels .cu (2-5s vs ~3min nvcc rebuild). Link
    // opcional de libnvrtc: sin toolkit el módulo degrada a error claro
    // (nvrtc.available()==false) — NUNCA rompe el build (patrón noop-stub).
    const nvrtc_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/nvrtc.zig"),
        .target = target,
        .optimize = optimize,
    });
    nvrtc_mod.link_libc = true;
    nvrtc_mod.addImport("cudaz", cudaz_mod); // lane-cuda UC-1
    if (has_cuda) nvrtc_mod.linkSystemLibrary("nvrtc", .{}); // lane-cuda UC-1.2 (debug import se añade post debug_mod, línea ~641)

    // === Lane D: bindings memoria externa (cuMemHostRegister) ===
    // === Módulo CUDA extern-mem bindings (lane-d D2) ===
    const cudaz_ext_mem_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/cudaz_ext_mem.zig"),
        .target = target,
        .optimize = optimize,
    });
    cudaz_ext_mem_mod.link_libc = true;

    // === Lane D: HostBank pin-after-fill (Contrato 5) ===
    cudaz_ext_mem_mod.addImport("cudaz", cudaz_mod);

    // === Módulo mem (lane-d: HostBank pin-after-fill) ===
    // MH-1 (CUDA): disk_tier para offload a disco
    const tier_mod = b.createModule(.{
        .root_source_file = b.path("src/mem/tier.zig"),
        .target = target,
        .optimize = optimize,
    });
    const tier_manager_mod = b.createModule(.{
        .root_source_file = b.path("src/mem/tier_manager.zig"),
        .target = target,
        .optimize = optimize,
    });
    const disk_tier_mod = b.createModule(.{
        .root_source_file = b.path("src/mem/disk_tier.zig"),
        .target = target,
        .optimize = optimize,
    });

    const host_bank_mod = b.createModule(.{
        .root_source_file = b.path("src/mem/host_bank.zig"),
        .target = target,
        .optimize = optimize,
    });
    host_bank_mod.addImport("cudaz", cudaz_mod);
    host_bank_mod.addImport("cudaz_ext_mem", cudaz_ext_mem_mod);
    host_bank_mod.addImport("disk_tier", disk_tier_mod); // MH-15: disk-backed host bank
    host_bank_mod.link_libc = true;

    // === Módulo backend_capabilities (lane-b1 Dev-B) ===
    const backend_capabilities_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/backend_capabilities.zig"),
        .target = target,
        .optimize = optimize,
    });
    backend_capabilities_mod.link_libc = true; // lane-b1 Dev-B: externs cu* locales

    // === Módulo CUDA Runtime API (patrón zig-cuda-agent) ===
    const cuda_runtime_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/cuda_runtime.zig"),
        .target = target,
        .optimize = optimize,
    });
    cuda_runtime_mod.link_libc = true;

    // === Módulo cublas (host) ===
    const cublas_mod = b.createModule(.{
        .root_source_file = b.path("src/matmul/cublas.zig"),
        .target = target,
        .optimize = optimize,
    });
    cublas_mod.link_libc = true;
    cublas_mod.addImport("core", core_mod);
    cublas_mod.addImport("cudaz", cudaz_mod);
    cublas_mod.addImport("time", time_mod);

    // === Módulo FP8 block-scaled kernels (lane-b) ===
    const fp8_kernels_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/fp8_block_kernels.zig"),
        .target = target,
        .optimize = optimize,
    });
    fp8_kernels_mod.addImport("cudaz", cudaz_mod);
    fp8_kernels_mod.addOptions("build_options", fp8_options);
    // debug_mod added after it's declared

    // === Módulo graph_capture (Phase 3: captura CUDA Graph reutilizable) ===
    // Módulo mínimo (cudaz + debug) para que lo compongan matmul y decode_graph.
    const graph_capture_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/graph_capture.zig"),
        .target = target,
        .optimize = optimize,
    });
    graph_capture_mod.addImport("cudaz", cudaz_mod);
    // graph_capture_mod: debug import added after debug_mod declaration.

    const matmul_mod = b.createModule(.{
        .root_source_file = b.path("src/matmul/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    matmul_mod.addImport("core", core_mod);
    matmul_mod.addImport("time", time_mod);
    matmul_mod.addImport("cudaz", cudaz_mod);
    matmul_mod.addImport("cublas", cublas_mod);
    matmul_mod.addImport("fp8_kernels", fp8_kernels_mod);
    matmul_mod.addImport("graph_capture", graph_capture_mod); // lane-f Phase 3
    matmul_mod.addOptions("build_options", options);

    // === Módulo dequant GPU de tensores GGUF ===
    const gguf_dequant_mod = b.createModule(.{
        .root_source_file = b.path("src/loader/gguf_dequant_gpu.zig"),
        .target = target,
        .optimize = optimize,
    });
    gguf_dequant_mod.link_libc = true;
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
    // === Módulo rope === (declarado aquí — KT-B: kv_cache lo importa)
    const rope_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/rope.zig"),
        .target = target,
        .optimize = optimize,
    });
    rope_mod.addImport("core", core_mod);

    const kv_cache_mod = b.createModule(.{
        .root_source_file = b.path("src/kv_cache.zig"),
        .target = target,
        .optimize = optimize,
    });
    kv_cache_mod.addImport("core", core_mod);
    kv_cache_mod.addImport("cudaz", cudaz_mod);
    // lane-f: gpu_dequant/offload usan la Runtime API; import con nombre para
    // evitar el conflicto de módulos (mismo archivo en 'cuda_runtime' y 'kv_cache').
    kv_cache_mod.addImport("cuda_runtime", cuda_runtime_mod);
    // KT-B (lane-f): kt_transfer — rope (strip/re-rope on-slice), matmul
    // (gemm f32 del mapper dense), debug (breadcrumbs [kt_transfer]).
    // rope se declara justo arriba; matmul (484) existe ya. Los imports
    // matmul/debug se añaden tras sus declaraciones (ver addImport-ktb abajo).

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

    // (rope_mod declarado arriba junto a kv_cache_mod — KT-B lo requiere)

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
    quant_weight_mod.addImport("core", core_mod);
    embedding_mod.addImport("quant_weight", quant_weight_mod); // lane-b1 7.1d: embeddingLookupQuant
    embedding_mod.addImport("gguf", gguf_mod); // lane-b1 7.1d: dequantBlock por fila

    // === Módulo debug (breadcrumbs de diagnóstico centralizados) ===
    const debug_mod = b.createModule(.{
        .root_source_file = b.path("src/debug.zig"),
        .target = target,
        .optimize = optimize,
    });
    resources_mod.addImport("debug", debug_mod); // coordinador: breadcrumb [cpu_budget]
    nvrtc_mod.addImport("debug", debug_mod); // lane-cuda UC-1: breadcrumbs [nvrtc]
    host_bank_mod.addImport("debug", debug_mod); // A3: breadcrumbs [pool]
    gguf_mod.addImport("debug", debug_mod); // A3: breadcrumbs [loader]
    cudaz_ext_mem_mod.addImport("debug", debug_mod); // A3: breadcrumbs [gpu_kernels]
    nvrtc_mod.addImport("time", time_mod); // lane-cuda UC-1.3: timing JIT en breadcrumb

    // === lane-cuda UC-3/UC-5/UC-2.2: utilidades CUDA (ficheros NUEVOS) ===
    const launch_mod = b.createModule(.{ // UC-5: LaunchConfig (puro)
        .root_source_file = b.path("src/cuda/launch.zig"),
        .target = target,
        .optimize = optimize,
    });
    const bridge_mod = b.createModule(.{ // UC-3: bridge type-safe
        .root_source_file = b.path("src/cuda/bridge.zig"),
        .target = target,
        .optimize = optimize,
    });
    bridge_mod.addImport("cudaz", cudaz_mod);
    const error_flag_mod = b.createModule(.{ // UC-2.2: wrapper host ErrorFlag
        .root_source_file = b.path("src/cuda/error_flag.zig"),
        .target = target,
        .optimize = optimize,
    });
    error_flag_mod.addImport("cudaz", cudaz_mod);
    error_flag_mod.addImport("debug", debug_mod);
    const nvtx_mod = b.createModule(.{ // UC-4.1: NVTX ranges (dlopen puro, sin link)
        .root_source_file = b.path("src/cuda/nvtx.zig"),
        .target = target,
        .optimize = optimize,
    });
    nvtx_mod.link_libc = true; // dlopen/dlsym

    // Paso LIGERO (sin cubins nvcc): verificación de las utilidades en
    // máquinas con disco justo. Zig: `zig build test-cuda-utils`.
    {
        const u_mod = b.createModule(.{
            .root_source_file = b.path("tests/test_cuda_utils.zig"),
            .target = target,
            .optimize = optimize,
        });
        u_mod.link_libc = true;
        u_mod.addImport("launch", launch_mod);
        u_mod.addImport("bridge", bridge_mod);
        u_mod.addImport("error_flag", error_flag_mod);
        u_mod.addImport("nvtx", nvtx_mod); // UC-4: fail-safe dlopen, sin GPU
        u_mod.addImport("cudaz", cudaz_mod);
        if (has_cuda) {
            u_mod.linkSystemLibrary("cuda", .{});
            u_mod.linkSystemLibrary("cudart", .{});
        } else {
            u_mod.addCSourceFile(.{ .file = b.path("src/cuda/cuda_noop_stub.c"), .flags = &.{} });
        }
        const u_t = b.addTest(.{ .root_module = u_mod, .filters = test_filters });
        const run_u = b.addRunArtifact(u_t);
        const cu_step = b.step("test-cuda-utils", "lane-cuda: LaunchConfig/bridge/ErrorFlag/NVTX unit tests (no cubins)");
        cu_step.dependOn(&run_u.step);
    }

    // KT-B (lane-f): kt_transfer (kv_cache) importa rope/matmul/debug —
    // declarados todos ya en este punto (rope 522, matmul 484, debug ↑).
    kv_cache_mod.addImport("rope", rope_mod);
    kv_cache_mod.addImport("matmul", matmul_mod);
    kv_cache_mod.addImport("debug", debug_mod);

    // === Módulo gguf_model ===
    const gguf_model_mod = b.createModule(.{
        .root_source_file = b.path("src/loader/gguf_model.zig"),
        .target = target,
        .optimize = optimize,
    });
    gguf_model_mod.addImport("debug", debug_mod); // lane-f F-3: breadcrumbs de carga
    gguf_model_mod.addImport("gguf", gguf_mod);
    gguf_model_mod.addImport("model_config", model_config_mod);
    gguf_model_mod.addImport("core", core_mod);
    gguf_model_mod.addImport("quant_weight", quant_weight_mod);
    gguf_model_mod.addImport("kv_cache", kv_cache_mod); // lane-f F2A: LMQ40 decode/QuantFormat

    // Lane D: matmul necesita breadcrumbs en el cache cuantizado residente.
    matmul_mod.addImport("debug", debug_mod);
    // Breadcrumbs también para graph_capture (DUMP_GRAPH).
    graph_capture_mod.addImport("debug", debug_mod);
    // Breadcrumbs para backend_capabilities (counters gated printLevel). lane-b1 Dev-B
    backend_capabilities_mod.addImport("debug", debug_mod);
    // Breadcrumbs disponibles también para el wrapper driver (cudaz).
    matmul_mod.addImport("debug", debug_mod); // lane-d D3: breadcrumbs gated
    matmul_mod.addImport("resources", resources_mod); // coordinador: budget parallel (post-freeze)
    matmul_mod.addImport("cudaz_ext_mem", cudaz_ext_mem_mod); // lane-d D5: streamWaitEvent
    cudaz_mod.addImport("debug", debug_mod);
    // Breadcrumbs disponibles también para el dequant GPU (lane-c C0).
    gguf_dequant_mod.addImport("debug", debug_mod);
    // FP8 block kernels debug
    fp8_kernels_mod.addImport("debug", debug_mod);

    // === Módulo mmproj_config (mmproj: parse clip.* metadata) ===
    const mmproj_config_mod = b.createModule(.{
        .root_source_file = b.path("src/loader/mmproj_config.zig"),
        .target = target,
        .optimize = optimize,
    });
    mmproj_config_mod.addImport("gguf", gguf_mod); // mmproj

    // === Módulo mmproj_model (mmproj: loader dual-GGUF) ===
    const mmproj_model_mod = b.createModule(.{
        .root_source_file = b.path("src/loader/mmproj_model.zig"),
        .target = target,
        .optimize = optimize,
    });
    mmproj_model_mod.addImport("gguf", gguf_mod); // mmproj
    mmproj_model_mod.addImport("mmproj_config", mmproj_config_mod); // mmproj
    mmproj_model_mod.addImport("core", core_mod); // mmproj
    mmproj_model_mod.addImport("quant_weight", quant_weight_mod); // mmproj
    mmproj_model_mod.addImport("debug", debug_mod); // mmproj

    // === Módulo vision: preprocess (stb_image + smartResize + normalize) ===
    const vision_preprocess_mod = b.createModule(.{
        .root_source_file = b.path("src/vision/preprocess.zig"),
        .target = target,
        .optimize = optimize,
    });
    vision_preprocess_mod.addImport("debug", debug_mod); // mmproj
    vision_preprocess_mod.link_libc = true; // mmproj: stb_image
    vision_preprocess_mod.addIncludePath(.{ .cwd_relative = "vendor" }); // mmproj: stb_image.h
    vision_preprocess_mod.addCSourceFile(.{ .file = b.path("vendor/stb_image.c"), .flags = &.{"-O2"} }); // mmproj

    // === Módulo vision: mrope_vision (M-RoPE VISION interleaved per-token) ===
    const mrope_vision_mod = b.createModule(.{
        .root_source_file = b.path("src/vision/mrope_vision.zig"),
        .target = target,
        .optimize = optimize,
    });
    mrope_vision_mod.addImport("core", core_mod); // mmproj

    // === Módulo vision: conv2d (im2col + GEMM) ===
    const vision_conv2d_mod = b.createModule(.{
        .root_source_file = b.path("src/vision/conv2d.zig"),
        .target = target,
        .optimize = optimize,
    });
    vision_conv2d_mod.addImport("core", core_mod); // mmproj
    vision_conv2d_mod.addImport("matmul", matmul_mod); // mmproj
    vision_conv2d_mod.addImport("debug", debug_mod); // mmproj

    // === Módulo vision: clip_attention (QKV fused + MRoPE + MHSA) ===
    const clip_attention_mod = b.createModule(.{
        .root_source_file = b.path("src/vision/clip_attention.zig"),
        .target = target,
        .optimize = optimize,
    });
    clip_attention_mod.addImport("core", core_mod); // mmproj
    clip_attention_mod.addImport("matmul", matmul_mod); // mmproj
    clip_attention_mod.addImport("quant_weight", quant_weight_mod); // mmproj
    clip_attention_mod.addImport("mrope_vision", mrope_vision_mod); // mmproj
    clip_attention_mod.addImport("mmproj_config", mmproj_config_mod); // mmproj
    clip_attention_mod.addImport("gguf", gguf_mod); // mmproj
    clip_attention_mod.addImport("time", time_mod); // mmproj
    clip_attention_mod.addImport("debug", debug_mod); // mmproj
    clip_attention_mod.addImport("resources", resources_mod); // coordinador: budget softmax ViT (post-freeze)

    // === Módulo vision: clip_block (LN→Attn→LN→FFN + deepstack) ===
    const clip_block_mod = b.createModule(.{
        .root_source_file = b.path("src/vision/clip_block.zig"),
        .target = target,
        .optimize = optimize,
    });
    clip_block_mod.addImport("core", core_mod); // mmproj
    clip_block_mod.addImport("matmul", matmul_mod); // mmproj
    clip_block_mod.addImport("norm", norm_mod); // mmproj
    clip_block_mod.addImport("quant_weight", quant_weight_mod); // mmproj
    clip_block_mod.addImport("clip_attention", clip_attention_mod); // mmproj
    clip_block_mod.addImport("mmproj_config", mmproj_config_mod); // mmproj
    clip_block_mod.addImport("gguf", gguf_mod); // mmproj (dequantF32Slice)
    clip_block_mod.addImport("debug", debug_mod); // mmproj

    // === Módulo vision: projector qwen3vl (merger mm.0/mm.2 + deepstack) ===
    const vision_qwen3vl_mod = b.createModule(.{
        .root_source_file = b.path("src/vision/projectors/qwen3vl.zig"),
        .target = target,
        .optimize = optimize,
    });
    vision_qwen3vl_mod.addImport("core", core_mod); // mmproj
    vision_qwen3vl_mod.addImport("matmul", matmul_mod); // mmproj
    vision_qwen3vl_mod.addImport("norm", norm_mod); // mmproj
    vision_qwen3vl_mod.addImport("quant_weight", quant_weight_mod); // mmproj
    vision_qwen3vl_mod.addImport("clip_attention", clip_attention_mod); // mmproj
    vision_qwen3vl_mod.addImport("clip_block", clip_block_mod); // mmproj
    vision_qwen3vl_mod.addImport("gguf", gguf_mod); // mmproj
    vision_qwen3vl_mod.addImport("debug", debug_mod); // mmproj

    // === Módulo vision: clip_encoder (orquestador ViT + spatial merge + pos_embd) ===
    const vision_clip_encoder_mod = b.createModule(.{
        .root_source_file = b.path("src/vision/clip_encoder.zig"),
        .target = target,
        .optimize = optimize,
    });
    vision_clip_encoder_mod.addImport("core", core_mod); // mmproj
    vision_clip_encoder_mod.addImport("matmul", matmul_mod); // mmproj
    vision_clip_encoder_mod.addImport("gguf", gguf_mod); // mmproj
    vision_clip_encoder_mod.addImport("mmproj_config", mmproj_config_mod); // mmproj
    vision_clip_encoder_mod.addImport("mmproj_model", mmproj_model_mod); // mmproj
    vision_clip_encoder_mod.addImport("quant_weight", quant_weight_mod); // mmproj
    vision_clip_encoder_mod.addImport("norm", norm_mod); // mmproj
    vision_clip_encoder_mod.addImport("conv2d", vision_conv2d_mod); // mmproj
    vision_clip_encoder_mod.addImport("clip_block", clip_block_mod); // mmproj
    vision_clip_encoder_mod.addImport("clip_attention", clip_attention_mod); // mmproj
    vision_clip_encoder_mod.addImport("mrope_vision", mrope_vision_mod); // mmproj
    vision_clip_encoder_mod.addImport("time", time_mod); // mmproj
    vision_clip_encoder_mod.addImport("qwen3vl_projector", vision_qwen3vl_mod); // mmproj
    vision_clip_encoder_mod.addImport("preprocess", vision_preprocess_mod); // mmproj
    vision_clip_encoder_mod.addImport("cudaz", cudaz_mod); // mmproj
    vision_clip_encoder_mod.addImport("debug", debug_mod); // mmproj

    // === Módulo vision: video (ffmpeg/ffprobe subprocess, TODO 10.7) ===
    const vision_video_mod = b.createModule(.{
        .root_source_file = b.path("src/vision/video.zig"),
        .target = target,
        .optimize = optimize,
    });
    vision_video_mod.addImport("debug", debug_mod); // mmproj 10.7
    vision_video_mod.link_libc = true; // mmproj 10.7: std.process.run

    // === Módulo vision: token_inject (embeds mixtos + pos-ids 2D) ===
    const vision_token_inject_mod = b.createModule(.{
        .root_source_file = b.path("src/vision/token_inject.zig"),
        .target = target,
        .optimize = optimize,
    });
    vision_token_inject_mod.addImport("core", core_mod); // mmproj
    vision_token_inject_mod.addImport("debug", debug_mod); // mmproj

    // === Módulo moe_cpu_gemv (Lane F1: GEMV cuantizado CPU) ===
    const moe_gemv_mod = b.createModule(.{
        .root_source_file = b.path("src/moe/cpu_gemv.zig"),
        .target = target,
        .optimize = optimize,
    });
    moe_gemv_mod.addImport("debug", debug_mod);
    moe_gemv_mod.addImport("kv_cache", kv_cache_mod);

    // === Módulo moe_cpu_executor (Lane F2: pool workers por núcleo físico) ===
    const moe_cpu_executor_mod = b.createModule(.{
        .root_source_file = b.path("src/moe/cpu_executor.zig"),
        .target = target,
        .optimize = optimize,
    });
    moe_cpu_executor_mod.addImport("debug", debug_mod);
    moe_cpu_executor_mod.addImport("resources", resources_mod); // coordinador: topología movida (post-freeze)
    moe_cpu_executor_mod.addImport("moe_cpu_gemv", moe_gemv_mod);
    moe_cpu_executor_mod.addImport("cudaz", cudaz_mod);

    // === Módulo cudaz_ext_sync (Lane F3: memops front-end CUDA) ===
    const ext_sync_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/cudaz_ext_sync.zig"),
        .target = target,
        .optimize = optimize,
    });
    ext_sync_mod.addImport("debug", debug_mod);
    moe_cpu_executor_mod.addImport("cudaz_ext_sync", ext_sync_mod);
    host_bank_mod.addImport("cudaz_ext_sync", ext_sync_mod); // lane-e: import que usaba host_bank.zig:9 (cudaHostAlloc memops)

    // === Módulo layer_kernels (elementwise GPU para capa híbrida residente) ===
    const layer_kernels_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/layer_kernels.zig"),
        .target = target,
        .optimize = optimize,
    });
    layer_kernels_mod.addImport("cudaz", cudaz_mod);
    layer_kernels_mod.addOptions("build_options", layer_options);
    layer_kernels_mod.addImport("debug", debug_mod);
    layer_kernels_mod.addImport("nvrtc", nvrtc_mod); // lane-cuda UC-1.3: JIT opt-in en loadModule
    layer_kernels_mod.addImport("bridge", bridge_mod); // lane-cuda UC-3.2: getT() typo-safe
    // === Módulo vision: clip_gpu (10.2 device-resident ViT) ===
    const clip_gpu_mod = b.createModule(.{
        .root_source_file = b.path("src/vision/clip_gpu.zig"),
        .target = target,
        .optimize = optimize,
    });
    clip_gpu_mod.addImport("core", core_mod); // mmproj
    clip_gpu_mod.addImport("cublas", cublas_mod); // mmproj
    clip_gpu_mod.addImport("cudaz", cudaz_mod); // mmproj
    clip_gpu_mod.addImport("layer_kernels", layer_kernels_mod); // mmproj
    clip_gpu_mod.addImport("mrope_vision", mrope_vision_mod); // mmproj
    clip_gpu_mod.addImport("debug", debug_mod); // mmproj
    // clip_encoder (definido antes) necesita clip_gpu — cableado post-def
    vision_clip_encoder_mod.addImport("clip_gpu", clip_gpu_mod); // mmproj

    // === Lane D: módulo stream_bench (harness standalone streaming denso) ===
    const stream_bench_mod = b.createModule(.{
        .root_source_file = b.path("examples/stream_bench.zig"),
        .target = target,
        .optimize = optimize,
    });
    stream_bench_mod.addImport("core", core_mod);
    stream_bench_mod.addImport("gguf", gguf_mod);
    stream_bench_mod.addImport("model_config", model_config_mod);
    stream_bench_mod.addImport("quant_weight", quant_weight_mod);
    stream_bench_mod.addImport("matmul", matmul_mod);
    stream_bench_mod.addImport("cublas", cublas_mod);
    stream_bench_mod.addImport("layer_kernels", layer_kernels_mod);
    stream_bench_mod.addImport("cudaz", cudaz_mod);
    stream_bench_mod.addImport("time", time_mod);
    stream_bench_mod.addImport("debug", debug_mod);
    stream_bench_mod.addImport("cudaz_ext_mem", cudaz_ext_mem_mod);
    // === Módulos lane-e (MoE offload) ===
    const gguf_moe_mod = b.createModule(.{
        .root_source_file = b.path("src/loader/gguf_moe.zig"),
        .target = target,
        .optimize = optimize,
    });
    gguf_moe_mod.addImport("gguf", gguf_mod);
    gguf_moe_mod.addImport("debug", debug_mod);

    const moe_offload_mod = b.createModule(.{
        .root_source_file = b.path("src/moe/offload_cache.zig"),
        .target = target,
        .optimize = optimize,
    });
    moe_offload_mod.addImport("debug", debug_mod);

    // === Módulo bandwidth (lane-f 4.7: q* auto-split, Gap 2 GPU↔CPU) ===
    const moe_bandwidth_mod = b.createModule(.{
        .root_source_file = b.path("src/moe/bandwidth.zig"),
        .target = target,
        .optimize = optimize,
    });
    moe_bandwidth_mod.addImport("debug", debug_mod);
    moe_offload_mod.addImport("bandwidth", moe_bandwidth_mod); // lane-f: q* en ensureExperts

    // === Módulo expert_bundle (lane-e 11.2: bundle contiguo PowerInfer) ===
    const moe_bundle_mod = b.createModule(.{
        .root_source_file = b.path("src/moe/expert_bundle.zig"),
        .target = target,
        .optimize = optimize,
    });
    moe_bundle_mod.addImport("debug", debug_mod); // lane-e 11.2
    moe_bundle_mod.addImport("gguf", gguf_mod); // lane-e 11.2: validateAgainstGguf
    moe_bundle_mod.addImport("gguf_moe", gguf_moe_mod); // lane-e 11.2: BankKind/layerSpec

    // === Módulo moe_hybrid ===
    const moe_hybrid_mod = b.createModule(.{
        .root_source_file = b.path("src/moe/hybrid.zig"),
        .target = target,
        .optimize = optimize,
    });
    moe_hybrid_mod.addImport("debug", debug_mod);
    moe_hybrid_mod.addImport("cuda_runtime", cuda_runtime_mod);
    const moe_cache_mod = b.createModule(.{
        .root_source_file = b.path("src/moe/cache.zig"),
        .target = target,
        .optimize = optimize,
    });
    moe_cache_mod.addImport("debug", debug_mod);
    const moe_cuda_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/moe_kernels.zig"),
        .target = target,
        .optimize = optimize,
    });
    moe_cuda_mod.addImport("cudaz", cudaz_mod);
    moe_cuda_mod.addOptions("build_options", moe_options);
    moe_cuda_mod.addImport("debug", debug_mod);
    moe_cuda_mod.addImport("offload_cache", moe_offload_mod);
    // T1-rescate lane-e: cudaz_ext_mem_mod + host_bank_mod ya declarados arriba
    // (junto al stub cudaz). Aquí solo el cableado que faltaba (lane-f):
    moe_cpu_executor_mod.addImport("cudaz_ext_mem", cudaz_ext_mem_mod);
    // lane-f Phase 3: presupuesto elástico (módulo puro, sin deps CUDA).
    const mem_budget_mod = b.createModule(.{
        .root_source_file = b.path("src/mem/budget.zig"),
        .target = target,
        .optimize = optimize,
    });
    mem_budget_mod.addImport("debug", debug_mod);
    // lane-e E2: OffloadCache como Consumer del presupuesto elástico + LFRU (MH-6).
    moe_offload_mod.addImport("budget", mem_budget_mod);
    moe_offload_mod.addImport("tier", tier_mod);
    moe_offload_mod.addImport("disk_tier", disk_tier_mod);
    kv_cache_mod.addImport("disk_tier", disk_tier_mod); // MH-8: kv_cache/offload usa disk_tier

    // lane-f Phase 3: loader FTW (conversión GGUF→FTW + read plan O_DIRECT).
    const ftw_mod = b.createModule(.{
        .root_source_file = b.path("src/loader/ftw.zig"),
        .target = target,
        .optimize = optimize,
    });
    ftw_mod.addImport("gguf", gguf_mod);
    ftw_mod.addImport("debug", debug_mod);
    // lane-e E2: HostBank.loadFromFtw (bulk load FTW → bancos pineables).
    host_bank_mod.addImport("ftw", ftw_mod);

    // (stream_bench_mod: ya declarado arriba junto a layer_kernels — Lane D)

    const moe_layer_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/moe_layer.zig"),
        .target = target,
        .optimize = optimize,
    });
    moe_layer_mod.addImport("gguf", gguf_mod);
    moe_layer_mod.addImport("cudaz", cudaz_mod);
    moe_layer_mod.addImport("layer_kernels", layer_kernels_mod);
    moe_layer_mod.addImport("moe_cuda", moe_cuda_mod);
    moe_layer_mod.addImport("offload_cache", moe_offload_mod);
    moe_layer_mod.addImport("gguf_moe", gguf_moe_mod);
    moe_layer_mod.addImport("host_bank", host_bank_mod);
    moe_layer_mod.addImport("moe_cpu_executor", moe_cpu_executor_mod);
    moe_layer_mod.addImport("debug", debug_mod);
    // lane-4.10: ExpertStreamer (streaming expert-por-experto 2-slot)
    const moe_expert_stream_mod = b.createModule(.{
        .root_source_file = b.path("src/moe/expert_streamer.zig"),
        .target = target,
        .optimize = optimize,
    });
    moe_expert_stream_mod.addImport("cudaz", cudaz_mod);
    moe_expert_stream_mod.addImport("debug", debug_mod);
    moe_expert_stream_mod.addImport("moe_cuda", moe_cuda_mod); // lane-4.10: FetchStream
    moe_layer_mod.addImport("expert_streamer", moe_expert_stream_mod); // lane-4.10

    // === Módulo decode_graph (CUDA Graphs para el decode por token) ===
    const decode_graph_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/decode_graph.zig"),
        .target = target,
        .optimize = optimize,
    });
    decode_graph_mod.addImport("cudaz", cudaz_mod);
    decode_graph_mod.addImport("debug", debug_mod);
    decode_graph_mod.addImport("graph_capture", graph_capture_mod); // lane-f P3
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

    // === Módulo short_conv ===
    const short_conv_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/short_conv.zig"),
        .target = target,
        .optimize = optimize,
    });
    short_conv_mod.addImport("core", core_mod);
    short_conv_mod.addImport("matmul", matmul_mod);
    short_conv_mod.addImport("cublas", cublas_mod);
    short_conv_mod.addImport("cudaz", cudaz_mod);
    short_conv_mod.addImport("layer_kernels", layer_kernels_mod);
    short_conv_mod.addImport("quant_weight", quant_weight_mod);
    short_conv_mod.addImport("gguf", gguf_mod);
    short_conv_mod.addImport("norm", norm_mod);
    short_conv_mod.addImport("debug", debug_mod);

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
    hybrid_attn_mod.addImport("fp8_kernels", fp8_kernels_mod);
    hybrid_attn_mod.addImport("debug", debug_mod);
    hybrid_attn_mod.addOptions("build_options", options);

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
    hybrid_layer_mod.addImport("short_conv", short_conv_mod);
    hybrid_layer_mod.addImport("cublas", cublas_mod);
    hybrid_layer_mod.addImport("cudaz", cudaz_mod);
    hybrid_layer_mod.addImport("layer_kernels", layer_kernels_mod);
    hybrid_layer_mod.addImport("fp8_kernels", fp8_kernels_mod);
    hybrid_layer_mod.addImport("debug", debug_mod);
    // lane-e E5 (fix restore de lane-c: faltaba este addImport — el módulo
    // existe en el archivo pero hybrid_layer no podía resolver @import)
    hybrid_layer_mod.addImport("moe_layer", moe_layer_mod);

    // === Módulo activation_pool (AirLLM activation memory reuse) ===
    const activation_pool_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/activation_pool.zig"),
        .target = target,
        .optimize = optimize,
    });
    activation_pool_mod.addImport("debug", debug_mod);

    hybrid_layer_mod.addImport("activation_pool", activation_pool_mod);

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

    // === Módulo engine_api metrics (minimal time wrapper for TUI) ===
    const engine_api_metrics_mod = b.createModule(.{
        .root_source_file = b.path("src/engine_api/metrics.zig"),
        .target = target,
        .optimize = optimize,
    });

    // === Módulo paged_attention ===
    const paged_attention_mod = b.createModule(.{
        .root_source_file = b.path("src/paged_attention/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    paged_attention_mod.addImport("cudaz", cudaz_mod);
    paged_attention_mod.addImport("kv_cache", kv_cache_mod);
    paged_attention_mod.addImport("debug", debug_mod);
    paged_attention_mod.addOptions("build_options", paged_options);

    // === Módulo kvarn_kernels (lane-b1 Dev A): espejo Zig + launchers C2 ===
    const kvarn_kernels_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/kvarn_kernels.zig"),
        .target = target,
        .optimize = optimize,
    });
    kvarn_kernels_mod.addImport("cudaz", cudaz_mod); // lane-b1
    kvarn_kernels_mod.addImport("kv_cache", kv_cache_mod); // lane-b1 (C1 de B2)
    kvarn_kernels_mod.addImport("debug", debug_mod); // lane-b1
    kvarn_kernels_mod.addOptions("build_options", kvarn_options); // lane-b1

    // === Módulo kvarn_gpu_cache (lane-b1 Dev-A M3): ownership device
    // por capa — records/stage/indices/descs. Vive junto a
    // kvarn_kernels (necesita sus Args; kv_cache NO puede importarlo
    // por ciclo de módulos). ===
    const kvarn_gpu_cache_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/kvarn_gpu_cache.zig"),
        .target = target,
        .optimize = optimize,
    });
    kvarn_gpu_cache_mod.addImport("cudaz", cudaz_mod); // lane-b1 Dev-A
    kvarn_gpu_cache_mod.addImport("kv_cache", kv_cache_mod); // lane-b1 Dev-A (C1)
    kvarn_gpu_cache_mod.addImport("kvarn_kernels", kvarn_kernels_mod); // lane-b1 Dev-A

    // === Módulo fattn_kvarn (lane-b1 Dev-B): dispatch + portable + vec ===
    const fattn_kvarn_mod = b.createModule(.{
        .root_source_file = b.path("src/cuda/fattn_kvarn.zig"),
        .target = target,
        .optimize = optimize,
    });
    fattn_kvarn_mod.addImport("cudaz", cudaz_mod); // lane-b1 Dev-B
    fattn_kvarn_mod.addImport("debug", debug_mod); // lane-b1 Dev-B
    fattn_kvarn_mod.addImport("backend_capabilities", backend_capabilities_mod); // lane-b1 Dev-B
    fattn_kvarn_mod.addImport("kvarn_kernels", kvarn_kernels_mod); // lane-b1 Dev-B
    fattn_kvarn_mod.addOptions("build_options", kvarn_options); // lane-b1 Dev-B

    // hybrid_attn usa PagedKVCache para el KV-cache de atención
    hybrid_attn_mod.addImport("paged_attention", paged_attention_mod);
    hybrid_layer_mod.addImport("paged_attention", paged_attention_mod);
    hybrid_attn_mod.addImport("kvarn_kernels", kvarn_kernels_mod); // lane-b 9.4: KvarnDesc size en FA-native hook
    hybrid_attn_mod.addImport("fattn_kvarn", fattn_kvarn_mod); // lane-b 9.4: portable FA launcher

    // === Lane A: bench_kv_quant (KV cuantizado memoria/tok-s) ===
    const bench_kv_quant_mod = b.createModule(.{
        .root_source_file = b.path("benchmarks/bench_kv_quant.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench_kv_quant_mod.addImport("paged_attention", paged_attention_mod);
    bench_kv_quant_mod.addImport("cudaz", cudaz_mod);
    bench_kv_quant_mod.addImport("layer_kernels", layer_kernels_mod);
    bench_kv_quant_mod.addImport("time", time_mod);
    bench_kv_quant_mod.addImport("debug", debug_mod);
    bench_kv_quant_mod.link_libc = true;
    if (has_cuda) {
        bench_kv_quant_mod.linkSystemLibrary("cuda", .{});
        bench_kv_quant_mod.linkSystemLibrary("cudart", .{});
        if (cuda_lib_dir_exists) bench_kv_quant_mod.addLibraryPath(.{ .cwd_relative = cuda_lib_path });
    } else {
        bench_kv_quant_mod.addCSourceFile(.{ .file = b.path("src/cuda/cuda_noop_stub.c"), .flags = &.{} });
    }

    // === Módulo transformer ===
    const transformer_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/layer.zig"),
        .target = target,
        .optimize = optimize,
    });
    transformer_mod.addImport("quant_weight", quant_weight_mod); // lane-b1 Dev-A: dequant on-the-fly legacy
    transformer_mod.addImport("debug", debug_mod); // lane-f 7.1c: breadcrumbs en path legacy
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
    transformer_mod.addImport("ssm", ssm_mod); // lane-f F1: ssmStageReport re-export
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
    pipeline_mod.addImport("norm", norm_mod); // lane-f 7.2: RMSNorm final pre-lm_head
    pipeline_mod.addImport("time", time_mod);
    pipeline_mod.addImport("nvtx", nvtx_mod); // lane-cuda UC-4.2: rangos nsys prefill/decode
    pipeline_mod.addImport("gguf", gguf_mod);
    pipeline_mod.addImport("model_config", model_config_mod);
    pipeline_mod.addImport("debug", debug_mod); // lane-rlt: RLT_CAPTURE_DIR breadcrumbs

    // === Módulo train (RLT adapter training) ===
    const train_rlt_mod = b.createModule(.{
        .root_source_file = b.path("src/train/rlt_layer.zig"),
        .target = target,
        .optimize = optimize,
    });
    const train_export_mod = b.createModule(.{
        .root_source_file = b.path("src/train/export_gguf.zig"),
        .target = target,
        .optimize = optimize,
    });
    const train_capture_mod = b.createModule(.{
        .root_source_file = b.path("src/train/capture.zig"),
        .target = target,
        .optimize = optimize,
    });
    const train_merge_mod = b.createModule(.{
        .root_source_file = b.path("src/train/merge_tool.zig"),
        .target = target,
        .optimize = optimize,
    });
    train_merge_mod.addImport("gguf", gguf_mod);
    pipeline_mod.addImport("capture", train_capture_mod);
    const train_mod = b.createModule(.{ .root_source_file = b.path("src/train/train.zig"), .target = target, .optimize = optimize });
    train_mod.addImport("rlt_layer", train_rlt_mod);
    train_mod.addImport("export_gguf", train_export_mod);
    train_mod.addImport("debug", debug_mod);

    // === Módulo speculative (lane-c C2: driver especulativo + sampler) ===
    const speculative_mod = b.createModule(.{
        .root_source_file = b.path("src/speculative/spec_driver.zig"),
        .target = target,
        .optimize = optimize,
    });
    speculative_mod.addImport("gguf", gguf_mod);
    speculative_mod.addImport("kv_cache", kv_cache_mod);
    speculative_mod.addImport("quant_weight", quant_weight_mod); // C4.1 MtpHead
    speculative_mod.addImport("cudaz", cudaz_mod);
    speculative_mod.addImport("debug", debug_mod);
    speculative_mod.addImport("layer_kernels", layer_kernels_mod); // C6.1 dflash encoder (lane-c 5.1): gather+GEMV+rmsNorm device
    // 5.2 (lane-b1): draft-model DFlash — dflash_draft.zig
    speculative_mod.addImport("gguf_model", gguf_model_mod); // lane-b1 5.2 dflash
    speculative_mod.addImport("model_config", model_config_mod); // lane-b1 5.2 dflash
    speculative_mod.addImport("core", core_mod); // lane-b1 5.2 dflash (Tensor)
    speculative_mod.addImport("hybrid_layer", hybrid_layer_mod); // lane-b1 5.2 dflash
    speculative_mod.addImport("paged_attention", paged_attention_mod); // lane-b1 5.2 dflash

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
    exe_mod.addImport("kvarn_gpu_cache", kvarn_gpu_cache_mod); // lane-b1 Dev-B (M3)
    exe_mod.addOptions("build_options", kvarn_options); // lane-b1 Dev-B (M3 slice 2: kvarn_cubin)
    // R0 release stamping: version + git sha (--version). Import propio
    // "version_info" — NO tocar build_options (es de kvarn).
    // 0.16 no tiene Run.captureStdout: sha via `git rev-parse` volcado a
    // fichero por addSystemCommand+addOutputFileArg no aplica a stdout;
    // en su lugar: override -Dgit-sha=<sha> del usuario del build, y si
    // no viene, "unknown" (tag final lleva el sha en el mensaje).
    const version_options = b.addOptions();
    version_options.addOption([]const u8, "version", "0.1.0");
    const git_sha = b.option([]const u8, "git-sha", "Git sha para --version (default: dev)") orelse "dev";
    version_options.addOption([]const u8, "git_sha", git_sha);
    exe_mod.addOptions("version_info", version_options);

    // === Módulo vram_budget (AirLLM VRAM budgeting) ===
    const vram_budget_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/vram_budget.zig"),
        .target = target,
        .optimize = optimize,
    });
    vram_budget_mod.addImport("debug", debug_mod);
    // lane-f P3: presupuesto elástico como fuente del techo de RebuildTx.
    vram_budget_mod.addImport("budget", mem_budget_mod);
    transformer_mod.addImport("vram_budget", vram_budget_mod);
    exe_mod.addImport("vram_budget", vram_budget_mod);
    hybrid_layer_mod.addImport("vram_budget", vram_budget_mod);

    // === Módulo gpu_weight_pool (GPU weight pool for layer streaming) ===
    const gpu_weight_pool_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/gpu_weight_pool.zig"),
        .target = target,
        .optimize = optimize,
    });
    gpu_weight_pool_mod.addImport("matmul", matmul_mod);
    gpu_weight_pool_mod.addImport("hybrid_layer", hybrid_layer_mod);
    gpu_weight_pool_mod.addImport("vram_budget", vram_budget_mod);
    gpu_weight_pool_mod.addImport("cudaz", cudaz_mod);
    gpu_weight_pool_mod.addImport("cublas", cublas_mod);
    gpu_weight_pool_mod.addImport("debug", debug_mod);
    gpu_weight_pool_mod.addImport("core", core_mod);
    transformer_mod.addImport("gpu_weight_pool", gpu_weight_pool_mod);
    exe_mod.addImport("gpu_weight_pool", gpu_weight_pool_mod);
    hybrid_layer_mod.addImport("gpu_weight_pool", gpu_weight_pool_mod);

    // === Módulo layer_streamer (AirLLM layer streaming) ===
    const layer_streamer_mod = b.createModule(.{
        .root_source_file = b.path("src/transformer/layer_streamer.zig"),
        .target = target,
        .optimize = optimize,
    });
    layer_streamer_mod.addImport("hybrid_layer", hybrid_layer_mod);
    layer_streamer_mod.addImport("gguf", gguf_mod);
    layer_streamer_mod.addImport("model_config", model_config_mod);
    layer_streamer_mod.addImport("matmul", matmul_mod);
    layer_streamer_mod.addImport("paged_attention", paged_attention_mod);
    layer_streamer_mod.addImport("core", core_mod);
    layer_streamer_mod.addImport("debug", debug_mod);
    layer_streamer_mod.addImport("vram_budget", vram_budget_mod);
    layer_streamer_mod.addImport("gpu_weight_pool", gpu_weight_pool_mod);
    layer_streamer_mod.addImport("cudaz", cudaz_mod);
    transformer_mod.addImport("layer_streamer", layer_streamer_mod);
    exe_mod.addImport("layer_streamer", layer_streamer_mod);
    layer_streamer_mod.addImport("gpu_weight_pool", gpu_weight_pool_mod);

    // Engine-side integration contract (MetricHooks, LayerMetrics, ...).
    const engine_api_contract_mod = b.createModule(.{
        .root_source_file = b.path("src/engine_api/contract.zig"),
        .target = target,
        .optimize = optimize,
    });
    const engine_api_mod = b.createModule(.{
        .root_source_file = b.path("src/engine_api/bridge.zig"),
        .target = target,
        .optimize = optimize,
    });
    engine_api_mod.addImport("contract", engine_api_contract_mod);
    engine_api_mod.addImport("time", time_mod);
    engine_api_mod.addImport("metrics", engine_api_metrics_mod);

    // === Módulo inference (server F2 T1: runHybridInference extraída) ===
    // Mismo grafo de imports que tenía main.zig para el camino de inferencia.
    // token_sink va ANTES: inference y server lo comparten (T2c).
    const server_token_sink_mod = b.createModule(.{
        .root_source_file = b.path("src/server/token_sink.zig"),
        .target = target,
        .optimize = optimize,
    }); // lane-server-f2
    const inference_mod = b.createModule(.{
        .root_source_file = b.path("src/inference/cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    inference_mod.addImport("core", core_mod); // lane-server-f2
    inference_mod.addImport("matmul", matmul_mod); // lane-server-f2
    inference_mod.addImport("fa", fa_mod); // lane-server-f2
    inference_mod.addImport("transformer", transformer_mod); // lane-server-f2
    inference_mod.addImport("kv_cache", kv_cache_mod); // lane-server-f2
    inference_mod.addImport("pipeline", pipeline_mod); // lane-server-f2
    inference_mod.addImport("gguf_model", gguf_model_mod); // lane-server-f2
    inference_mod.addImport("quant_weight", quant_weight_mod); // lane-f 7.1d wiring: emb cuant-residente (Emb.lookup)
    inference_mod.addImport("gguf_tokenizer", gguf_tokenizer_mod); // lane-server-f2
    inference_mod.addImport("tokenizer", tokenizer_mod); // lane-server-f2
    inference_mod.addImport("cudaz", cudaz_mod); // lane-server-f2
    inference_mod.addImport("cublas", cublas_mod); // lane-server-f2
    inference_mod.addImport("layer_kernels", layer_kernels_mod); // lane-server-f2
    inference_mod.addImport("ssm", ssm_mod); // lane-f (7.1-fix: qgemmTypeFor en estimador host)
    inference_mod.addImport("gguf", gguf_mod); // lane-server-f2
    inference_mod.addImport("embedding", embedding_mod); // lane-server-f2
    inference_mod.addImport("norm", norm_mod); // lane-server-f2
    inference_mod.addImport("paged_attention", paged_attention_mod); // lane-server-f2
    inference_mod.addImport("decode_graph", decode_graph_mod); // lane-server-f2
    inference_mod.addImport("layer_streamer", layer_streamer_mod); // lane-server-f2
    inference_mod.addImport("vram_budget", vram_budget_mod); // lane-server-f2
    inference_mod.addImport("model_config", model_config_mod); // lane-server-f2
    inference_mod.addImport("time", time_mod); // lane-server-f2
    inference_mod.addImport("resources", resources_mod); // coordinador: cpuLmHeadLogits budget (post-freeze)
    inference_mod.addImport("debug", debug_mod); // lane-server-f2
    inference_mod.addImport("speculative", speculative_mod); // lane-server-f2
    inference_mod.addImport("gguf_moe", gguf_moe_mod); // lane-server-f2
    inference_mod.addImport("moe_layer", moe_layer_mod); // lane-server-f2
    inference_mod.addImport("moe_cuda", moe_cuda_mod); // lane-server-f2
    inference_mod.addImport("offload_cache", moe_offload_mod); // lane-server-f2
    inference_mod.addImport("host_bank", host_bank_mod); // lane-server-f2
    inference_mod.addImport("expert_bundle", moe_bundle_mod); // lane-e 11.2: bundle contiguo
    inference_mod.addImport("cpu_executor", moe_cpu_executor_mod); // lane-server-f2
    inference_mod.addImport("moe_cpu_gemv", moe_gemv_mod); // lane-server-f2
    inference_mod.addImport("token_sink", server_token_sink_mod); // lane-server-f2
    // lane-f P3: options propias del inference (kvarn_cubin) — reusar
    // kvarn_options crea módulo duplicado en el grafo (error build_options3).
    const inference_options = b.addOptions();
    inference_options.addOption([]const u8, "kvarn_cubin", if (kvarn_cubin != null)
        b.getInstallPath(.{ .custom = "lib" }, "kvarn_kernels.cubin")
    else
        "");
    // lane-b 9.4: FA-native necesita el portable cubin (d128/d256).
    inference_options.addOption([]const u8, "fattn_cubin", if (fattn_kvarn_portable_cubin != null)
        b.getInstallPath(.{ .custom = "lib" }, "fattn_kvarn_portable.cubin")
    else
        "");
    inference_mod.addOptions("build_options", inference_options); // lane-f P3: kvarn M3 (trunk)
    inference_mod.addImport("kvarn_gpu_cache", kvarn_gpu_cache_mod); // lane-f P3: kvarn M3
    inference_mod.addImport("mmproj_model", mmproj_model_mod); // lane-f P3: vision multi-imagen
    inference_mod.addImport("mmproj_config", mmproj_config_mod); // lane-f P3
    inference_mod.addImport("vision_preprocess", vision_preprocess_mod); // lane-f P3
    inference_mod.addImport("vision_clip_encoder", vision_clip_encoder_mod); // lane-f P3
    inference_mod.addImport("vision_clip_gpu", clip_gpu_mod); // lane-f P3
    inference_mod.addImport("vision_token_inject", vision_token_inject_mod); // lane-f P3
    inference_mod.addImport("vision_video", vision_video_mod); // lane-mmproj 10.7: --video en encodeVision (cli)
    exe_mod.addImport("inference", inference_mod); // lane-server-f2

    // === Dep httpx.zig (HTTP/1.1 server para OpenAI/Anthropic/Ollama compat) ===
    const httpx_dep = b.dependency("httpx", .{
        .target = target,
        .optimize = optimize,
    });
    const httpx_mod = httpx_dep.module("httpx");

    // === Módulos del servidor HTTP (src/server/*.zig) ===
    const server_engine_mod = b.createModule(.{
        .root_source_file = b.path("src/server/engine.zig"),
        .target = target,
        .optimize = optimize,
    });
    const server_sse_mod = b.createModule(.{
        .root_source_file = b.path("src/server/sse.zig"),
        .target = target,
        .optimize = optimize,
    });
    server_sse_mod.addImport("httpx", httpx_mod);
    const server_json_util_mod = b.createModule(.{
        .root_source_file = b.path("src/server/json_util.zig"),
        .target = target,
        .optimize = optimize,
    });
    const server_batching_mod = b.createModule(.{
        .root_source_file = b.path("src/server/batching.zig"),
        .target = target,
        .optimize = optimize,
    }); // lane-server-f2
    server_batching_mod.addImport("token_sink", server_token_sink_mod); // lane-server-f2
    server_batching_mod.addImport("debug", debug_mod); // lane-server-f2
    server_batching_mod.addImport("time", time_mod); // lane-server-f2
    const server_auth_mod = b.createModule(.{
        .root_source_file = b.path("src/server/auth.zig"),
        .target = target,
        .optimize = optimize,
    }); // lane-server-f2
    server_auth_mod.addImport("debug", debug_mod); // lane-server-f2
    const server_audit_mod = b.createModule(.{
        .root_source_file = b.path("src/server/audit.zig"),
        .target = target,
        .optimize = optimize,
    }); // lane-server-f2
    server_audit_mod.addImport("debug", debug_mod); // lane-server-f2
    server_audit_mod.addImport("time", time_mod); // lane-server-f2
    const server_validation_mod = b.createModule(.{
        .root_source_file = b.path("src/server/validation.zig"),
        .target = target,
        .optimize = optimize,
    }); // lane-server-f2
    const server_quite_mod = b.createModule(.{
        .root_source_file = b.path("src/server/quite.zig"),
        .target = target,
        .optimize = optimize,
    }); // lane-server-f2
    const server_chat_template_mod = b.createModule(.{
        .root_source_file = b.path("src/server/chat_template.zig"),
        .target = target,
        .optimize = optimize,
    });
    const server_openai_mod = b.createModule(.{
        .root_source_file = b.path("src/server/openai.zig"),
        .target = target,
        .optimize = optimize,
    });
    server_openai_mod.addImport("httpx", httpx_mod);
    server_openai_mod.addImport("sse", server_sse_mod);
    server_openai_mod.addImport("chat_template", server_chat_template_mod);
    server_openai_mod.addImport("engine", server_engine_mod);
    server_openai_mod.addImport("time", time_mod);
    server_openai_mod.addImport("json_util", server_json_util_mod);
    server_openai_mod.addImport("inference", inference_mod); // lane-server-f2
    server_openai_mod.addImport("tokenizer", tokenizer_mod); // lane-server-f2
    server_openai_mod.addImport("token_sink", server_token_sink_mod); // lane-server-f2
    server_openai_mod.addImport("validation", server_validation_mod); // lane-server-f2
    server_openai_mod.addImport("quite", server_quite_mod); // lane-server-f2
    server_openai_mod.addImport("gguf_model", gguf_model_mod); // lane-server-f2
    server_openai_mod.addImport("matmul", matmul_mod); // lane-server-f2
    server_openai_mod.addImport("cudaz", cudaz_mod); // lane-server-f2
    const server_anthropic_mod = b.createModule(.{
        .root_source_file = b.path("src/server/anthropic.zig"),
        .target = target,
        .optimize = optimize,
    });
    server_anthropic_mod.addImport("httpx", httpx_mod);
    server_anthropic_mod.addImport("sse", server_sse_mod);
    server_anthropic_mod.addImport("chat_template", server_chat_template_mod);
    server_anthropic_mod.addImport("engine", server_engine_mod);
    server_anthropic_mod.addImport("time", time_mod);
    server_anthropic_mod.addImport("json_util", server_json_util_mod);
    server_anthropic_mod.addImport("openai", server_openai_mod); // lane-server-f2 T6
    server_anthropic_mod.addImport("inference", inference_mod); // lane-server-f2 T6
    server_anthropic_mod.addImport("tokenizer", tokenizer_mod); // lane-server-f2 T6
    server_anthropic_mod.addImport("gguf_tokenizer", gguf_tokenizer_mod); // lane-server-f2 T6
    server_anthropic_mod.addImport("token_sink", server_token_sink_mod); // lane-server-f2 T6
    server_anthropic_mod.addImport("validation", server_validation_mod); // lane-server-f2 T6
    server_anthropic_mod.addImport("quiet", server_quite_mod); // lane-server-f2 T6
    const server_ollama_mod = b.createModule(.{
        .root_source_file = b.path("src/server/ollama.zig"),
        .target = target,
        .optimize = optimize,
    });
    server_ollama_mod.addImport("httpx", httpx_mod);
    server_ollama_mod.addImport("sse", server_sse_mod);
    server_ollama_mod.addImport("chat_template", server_chat_template_mod);
    server_ollama_mod.addImport("engine", server_engine_mod);
    server_ollama_mod.addImport("time", time_mod);
    server_ollama_mod.addImport("json_util", server_json_util_mod);
    server_ollama_mod.addImport("openai", server_openai_mod); // lane-server-f2 T7
    server_ollama_mod.addImport("inference", inference_mod); // lane-server-f2 T7
    server_ollama_mod.addImport("token_sink", server_token_sink_mod); // lane-server-f2 T7
    server_ollama_mod.addImport("validation", server_validation_mod); // lane-server-f2 T7
    const server_mod = b.createModule(.{
        .root_source_file = b.path("src/server/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    server_mod.addImport("httpx", httpx_mod);
    server_mod.addImport("sse", server_sse_mod);
    server_mod.addImport("openai", server_openai_mod);
    server_mod.addImport("anthropic", server_anthropic_mod);
    server_mod.addImport("ollama", server_ollama_mod);
    server_mod.addImport("engine", server_engine_mod);
    server_mod.addImport("auth", server_auth_mod); // lane-server-f2
    server_mod.addImport("audit", server_audit_mod); // lane-server-f2
    server_mod.addImport("validation", server_validation_mod); // lane-server-f2
    server_mod.addImport("inference", inference_mod); // lane-server-f2
    server_mod.addImport("token_sink", server_token_sink_mod); // lane-server-f2 T8 (warmup)
    server_mod.addImport("gguf_model", gguf_model_mod); // lane-server-f2
    server_mod.addImport("time", time_mod); // lane-server-f2
    server_mod.addImport("debug", debug_mod); // A3: breadcrumbs [server]
    server_engine_mod.addImport("debug", debug_mod);
    server_engine_mod.addImport("time", time_mod);
    server_engine_mod.addImport("gguf_model", gguf_model_mod);
    server_engine_mod.addImport("gguf_tokenizer", gguf_tokenizer_mod);
    server_engine_mod.addImport("tokenizer", tokenizer_mod);
    server_engine_mod.addImport("pipeline", pipeline_mod);
    server_engine_mod.addImport("engine_api", engine_api_mod);
    exe_mod.addImport("server", server_mod);

    // === Módulo activation_pool (AirLLM activation memory reuse) ===
    exe_mod.addImport("vram_budget", vram_budget_mod);

    exe_mod.addImport("cublas", cublas_mod);
    exe_mod.addImport("time", time_mod);
    exe_mod.addImport("debug", debug_mod);
    exe_mod.addImport("speculative", speculative_mod); // lane-c: driver especulativo
    exe_mod.addImport("engine_api", engine_api_mod);
    exe_mod.addImport("mmproj_config", mmproj_config_mod); // mmproj
    exe_mod.addImport("mmproj_model", mmproj_model_mod); // mmproj
    exe_mod.addImport("vision_preprocess", vision_preprocess_mod); // mmproj
    exe_mod.addImport("vision_clip_encoder", vision_clip_encoder_mod); // mmproj
    exe_mod.addImport("vision_clip_gpu", clip_gpu_mod); // mmproj
    exe_mod.addImport("vision_video", vision_video_mod); // mmproj 10.7: --video
    exe_mod.addImport("vision_token_inject", vision_token_inject_mod); // mmproj
    exe_mod.addImport("gguf_moe", gguf_moe_mod); // lane-f F/C: wiring MoE-aware
    exe_mod.addImport("moe_layer", moe_layer_mod); // lane-f F/C
    exe_mod.addImport("moe_cuda", moe_cuda_mod); // lane-f F/C
    exe_mod.addImport("offload_cache", moe_offload_mod); // lane-f F/C
    exe_mod.addImport("host_bank", host_bank_mod); // lane-f F/C: copy-once 4.3'
    exe_mod.addImport("cpu_executor", moe_cpu_executor_mod); // lane-f F/C: Contrato 8
    exe_mod.addImport("moe_cpu_gemv", moe_gemv_mod); // lane-f F/C
    // === Módulos lane-b3 (P4 wiring main.zig) ===
    // kld y presets viven en src/bench/ y src/cli/; los exponemos al
    // ejecutable para el subcomando `kld` y `--preset` (P3.4 C5). Solo
    // importan `std` y `debug` — sin dependencias CUDA — por eso la
    // creación va aquí, fuera del bloque condicional de test_files.
    const kld_mod = b.createModule(.{
        .root_source_file = b.path("src/bench/kld.zig"),
        .target = target,
        .optimize = optimize,
    });
    kld_mod.addImport("debug", debug_mod); // breadcrumb gated DUMP_KVARN
    exe_mod.addImport("kld", kld_mod);
    const presets_mod = b.createModule(.{
        .root_source_file = b.path("src/cli/presets.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe_mod.addImport("presets", presets_mod);

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
        // lane-f Phase 3: launchers encode MXFP4/Q8_0.
        if (quant_encode_obj) |obj| exe_mod.addObjectFile(obj);
    } else {
        // CI sin toolkit: no-ops para que los extern "c" de cudaz resuelvan.
        // cuInit→error ⇒ isCudaAvailable()=false ⇒ paths GPU se autodesactivan.
        exe_mod.addCSourceFile(.{ .file = b.path("src/cuda/cuda_noop_stub.c"), .flags = &.{} });
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
        "tests/test_kv_trace.zig", // lane-kvc paso-1 KV-Codec: captura trazas K/V
        "tests/test_kv_predictor.zig", // lane-kvc paso-1 KV-Codec: harness predictor + gate
        "tests/test_kv_sr.zig", // lane-f paso-0 KV-Codec: SR en V (sesgo 4× menor)
        "tests/test_resources.zig", // coordinador post-freeze: presupuesto CPU central
        "tests/test_rope_norm.zig", // lane-f F1v7 (11.6): pairing RoPE llama = NORM
        "tests/test_quant_weight_i2s.zig", // lane-kvc BitNet P3: dequant i2_s escala global
        "tests/test_matmul.zig",
        "tests/test_flash_attention.zig",
        "tests/test_online_softmax.zig",
        "tests/test_backend_capabilities.zig", // lane-b1 Dev-B
        "tests/test_kvarn_wht_gpu.zig", // lane-b1 Dev-A: gate F0 WHT smoke
        "tests/test_kvarn_d64_gpu.zig", // 9.12 (lane-cuda) F2: D64 GPU smoke
        "tests/test_kvarn_store_gpu.zig", // lane-b1 Dev-A: A8/M1 store bit-exacto
        "tests/test_kvarn_mma_gpu.zig", // lane-b1 Dev-A: A9 MMA smoke
        "tests/test_kvarn_split_gpu.zig", // lane-b1 Dev-A: A10 decode-split
        "tests/test_kvarn_init_descs_gpu.zig", // lane-b1 Dev-B: B3 init_descs
        "tests/test_kvarn_gpu_cache.zig", // lane-b1 Dev-A: M3 KvarnGpuCache surface
        "tests/test_kvarn_main_pattern_gpu.zig", // lane-b1 Dev-A: M3 main-pattern integration
        "tests/test_kvarn_iso_store_gpu.zig", // lane-b1 Dev-A: bisect hang store kvh=2
        "tests/test_fattn_kvarn_gpu.zig", // lane-b1 Dev-B: B5 portable FA
        "tests/test_fattn_kvarn_wht_gpu.zig", // lane-b1 Dev-B: B4/D3 WHT prep
        "tests/test_fattn_kvarn_vec_gpu.zig", // lane-b1 Dev-B: B6 vec
        "tests/test_fattn_kvarn_bench_gpu.zig", // lane-b1 Dev-B: B8 bench skeleton
        "tests/test_fattn_kvarn_m1_seed_gpu.zig", // lane-b1 Dev-B: B5 M1 1000-seed
        "tests/test_fattn_kvarn_d256_gpu.zig", // lane-b1 Dev-B: B4-iter5 D256 E2E
        "tests/test_fattn_kvarn_d512_gpu.zig", // lane-b1 Dev-B: iter-12 D512 E2E
        "tests/test_fattn_kvarn_multistream_gpu.zig", // lane-b1 Dev-B: iter-13 multistream matrix
        "tests/test_fattn_kvarn_d4_tail_gpu.zig", // lane-b1 Dev-B: B5-iter3 D6 materialize
        "tests/test_fattn_kvarn_smoke_gpu.zig", // lane-b1 Dev-B: iter-28 ARITY smoke + split launcher
        "tests/test_fattn_kvarn_repro_gpu.zig", // lane-b 9.4: repro E2E con dumps reales
        "tests/test_transformer.zig",
        "tests/test_embedding_quant.zig", // lane-b1 7.1d: paridad lookup cuant-residente
        "tests/test_lm_head_quant.zig", // lane-c 7.1d: paridad GEMV lm_head cuant-residente
        "tests/test_kv_cache.zig",
        "tests/test_kt_transfer.zig", // lane-f KT-B: KV-transfer gate-0 + formato .ktb
        "tests/test_kv_p03_types.zig", // lane-b2 B2-P0.3 q2_0s/q2_1/q3_0/q3_1/q6_0/q6_1 roundtrip
        "tests/test_kv_tail.zig", // lane-b2 B2-P0.2 KVCPT cola exacta + rollback
        "tests/test_kvarn_roundtrip.zig", // lane-b2 B2-P0.1 CPU reference
        "tests/test_kvarn_manager.zig", // lane-c 9.1 C-1: store KVarN manager + parse CLI
        "tests/test_kvarn_kld.zig", // lane-b2 B2-P0.1 KLD/calidad
        "tests/test_kvcpt_integration.zig", // lane-b2 B2-P0.2 KVCPT integration (WIP portado)
        "tests/test_paged_attention.zig",
        "tests/test_gguf.zig",
        "tests/test_u2_prefill_dense_gpu.zig", // lane-b1 U2: paridad prefill GPU dense batched (n>1)
        "tests/test_u2_kv_append_diag.zig", // lane-b1 U2-diag: kvAppend batched vs unrolled
        "tests/test_u2_prefill_llama_gpu.zig", // lane-b1 U2: gate 512tok capa real Llama (chasis U1)
        "tests/test_u2_llama_stack_gpu.zig", // lane-b1 U2: pila completa 28 capas + head, greedy top-1 (chasis U1)
        "tests/test_paged_attention_gpu.zig",
        "tests/test_dequant_gpu.zig",
        "tests/test_kv_append_quant_gpu.zig",
        "tests/test_fused_decode_gpu.zig",
        "tests/test_offload.zig",
        "tests/test_spec_sampler.zig", // lane-c C2
        "tests/test_paged_kv_truncate.zig", // lane-c C3
        "tests/test_spec_mtp_head.zig", // lane-c C4-tune T2 (SPEC_MTP_MODEL)
        "tests/test_dflash_draft.zig", // lane-b1 5.2 (DFlash kvInject+denoise CPU)
        "tests/test_dflash_e2e.zig", // lane-b1 5.2 (sidecar real: dual-load+kvInject+denoise)
        "tests/test_host_bank.zig", // lane-d D2 (HostBank pin-after-fill, Contrato 5)
        "tests/test_quant_resident.zig", // lane-d D7 (semántica claves host/dev + contadores)
        "tests/test_q4_packed.zig", // lane-f 1.3 (paridad GEMV packed dual-view + repack roundtrip)
        "tests/test_fp8_block.zig", // lane-f 2.1 (paridad FP8 block GEMM/GEMV vs CPU)
        "tests/test_moe_zerocopy_probe.zig", // lane-e: experimento MAP_SHARED // lane-c C4-tune T2 (SPEC_MTP_MODEL)
        "tests/test_cpu_gemv.zig", // lane-f F1
        "tests/test_moe_layer_gpu.zig", // lane-e E5 GPU
        "tests/test_moe_gather_gpu.zig", // lane-e E4 GPU
        "tests/test_moe_kernels_gpu.zig", // lane-e E3 GPU
        "tests/test_moe_cache.zig", // lane-e E1/E2
        "tests/test_moe_bandwidth.zig", // lane-f 4.7: q* auto-split
        "tests/test_moe_e2e.zig", // lane-e E2-e2e: executor-merge host
        "tests/test_moe_expert_stream_gpu.zig", // 4.10/4.12: ExpertStreamer ping-pong + resize elástico
        "tests/test_expert_bundle.zig", // lane-e 11.2: bundle contiguo integridad
        "tests/test_hybrid_sync.zig", // lane-f F3
        "tests/test_vram_rebuild.zig", // lane-f F5
        "tests/test_quant_encode_gpu.zig", // lane-f Phase 3 (paridad encode)
        "tests/test_argmax_gpu.zig", // lane-a G2/1.7 (paridad argmax device vs host + empates) — restaurada (perdida en 7a0b6d7)
        "tests/test_nvrtc_gpu.zig", // lane-cuda UC-1.5: paridad JIT NVRTC vs cubin + ErrorFlag smoke
        "tests/test_cuda_utils.zig", // lane-cuda UC-2/UC-3/UC-5: LaunchConfig/bridge/ErrorFlag (ligero)
        "tests/test_deltanet_warp.zig", // STUDY §5.6 (paridad warp ΔNet)
        "tests/test_deltanet_fused.zig", // STUDY §5.1 (paridad GDN fused)
        "tests/test_gdn_l2norm.zig", // 1.14 (lane-c): fórmula FLA l2norm Q/K + inline ssm tests
        "tests/test_rlt_feedback.zig", // RLT: gated merge CPU paridad + GPU paridad + alpha=0 identidad
        "tests/test_rlt_swa.zig", // RLT: SWA sliding window mask logic + attention divergence
        "tests/test_dp4a_parity_p05.zig", // IQ: dp4a parity iq3_s/iq2_s/iq4_xs (P0-5)
        "tests/test_dp4a_parity_p06.zig", // IQ: dp4a parity q4_1/q2_k/iq4_nl/iq3_xxs/iq2_xxs/iq2_xs (P0-6)
        "tests/test_rlt_prefill.zig", // RLT: recurrent prefill vs parallel parity (linear model)
        "tests/test_rlt_backward.zig", // RLT: finite-diff vs analytic gradient check
        "tests/test_rlt_train.zig", // RLT: smoke test (50 steps, loss decreases)
        "tests/test_rlt_export.zig", // RLT: export GGUF roundtrip vs parser del engine
        "tests/test_rlt_merge.zig", // RLT: merge sidecar→base byte-level + paridad datos
        "tests/test_rltcap.zig", // R-4 (RLT training Fase 2): readRltcap roundtrip .rltcap v1
        "tests/test_prefill_chunked.zig", // STUDY §5.2 — Dev C paridad chunked ΔNet prefill
        "tests/test_prefill_wy_gpu.zig", // STUDY §5.2 (1.4) — lane-b paridad kernels WY GPU vs per-token
        "tests/test_prefill_wy.zig", // STUDY §5.2 (1.4) — paridad oráculo WY-chunk vs per-token
        "tests/test_dflash_encoder.zig", // C6.1 (5.1, lane-c) — paridad DflashEncoder device (GGUF_MODEL_PATH + DFLASH_SIDECAR_PATH)
        //   de Dev C (compila con error c_int/c_uint en L247). Reactivar al aterrizar.
        "tests/test_study_suite.zig", // STUDY: suite compartida (coordinador)
        "tests/test_server_f2.zig", // lane-server-f2 (auth/validation/audit/batching/sink/quite)
        "tests/test_server_e2e.zig", // lane-server-f2 T9 (E2E; SERVER_TEST_MODEL para activar)
        // NOTA: pasos dedicados para study_suite están más abajo (§5.8 bench).
        "tests/test_adaptive_dm.zig", // lane-b3 P1.1
        "tests/test_loop_guard.zig", // lane-b3 P1.2
        "tests/test_loop_guard_false_positive.zig", // lane-b3 P1.2
        "src/bench/kld.zig", // lane-b3 P3.5 (tests sanity del núcleo puro)
        "tests/test_presets.zig", // lane-b3 P3.4 (C5: superficie flags)
        "src/engine_api/metrics.zig", // lane-b3 T5 (spec_* + route counters)
        "src/bench/adaptive_bench.zig", // lane-b3 T6 (micro-bench profit vs off)
        "src/bench/controller_overhead.zig", // lane-b3 T7 (ns/op del controller)
        "src/mem/host_bank.zig",
        "src/mem/budget.zig", // lane-f Phase 3 (elástico)
        "src/loader/ftw.zig", // lane-f Phase 3 (FTW)
        "src/cuda/graph_capture.zig", // lane-f Phase 3 (captura)
        "src/moe/hybrid.zig",
        "src/matmul/hybrid_exec.zig",
        "tests/test_mmproj.zig", // mmproj/vision (PLAN_MMPROJ)
        "tests/test_p0_6_dp4a.zig", // lane-cuda P0-6 dp4a parity
        "tests/test_dbg_scratch.zig",
        "tests/test_debug_scope.zig",
        "tests/test_dflash2_topk.zig", // lane-cuda 5.3 dflash2 selector top-K benchmark
        "tests/test_tier_manager.zig", // IQ MH-4: LFRU 3-tier admission
    };

    inline for (test_files) |tf| {
        const tmod = b.createModule(.{
            .root_source_file = b.path(tf),
            .target = target,
            .optimize = optimize,
        });
        tmod.link_libc = true;
        tmod.addImport("core", core_mod);
        tmod.addImport("matmul", matmul_mod);
        tmod.addImport("fa", fa_mod);
        tmod.addImport("transformer", transformer_mod);
        tmod.addImport("kv_cache", kv_cache_mod);
        tmod.addImport("gguf", gguf_mod); // lane-kvc P1: dequant i2_s test fixture
        tmod.addImport("cudaz", cudaz_mod);
        tmod.addImport("cudaz_ext_mem", cudaz_ext_mem_mod);
        tmod.addImport("cudaz_ext_mem", cudaz_ext_mem_mod);
        tmod.addImport("layer_kernels", layer_kernels_mod); // lane-f 1.3: q4 packed paridad
        tmod.addImport("nvrtc", nvrtc_mod); // lane-cuda UC-1.5: paridad JIT vs cubin
        tmod.addImport("launch", launch_mod); // lane-cuda UC-5
        tmod.addImport("bridge", bridge_mod); // lane-cuda UC-3
        tmod.addImport("error_flag", error_flag_mod); // lane-cuda UC-2.2
        tmod.addImport("nvtx", nvtx_mod); // lane-cuda UC-4: dlopen puro
        tmod.addImport("fp8_kernels", fp8_kernels_mod); // lane-f 2.1: quantizer FP8 directo
        tmod.addImport("cublas", cublas_mod); // lane-f 1.3: GpuTensor readback
        tmod.addImport("time", time_mod); // lane-f 1.3: (reserva bench)
        tmod.addImport("backend_capabilities", backend_capabilities_mod); // lane-b1 Dev-B
        if (std.mem.eql(u8, tf, "src/moe/hybrid.zig")) {
            tmod.addImport("moe_hybrid", moe_hybrid_mod);
            tmod.addImport("debug", debug_mod);
            tmod.addImport("cuda_runtime", cuda_runtime_mod);
            tmod.addOptions("build_options", options);
            if (hybrid_split_obj) |obj| {
                tmod.addObjectFile(obj);
            }
        }
        if (std.mem.eql(u8, tf, "src/mem/host_bank.zig")) {
            tmod.addImport("cudaz_ext_mem", cudaz_ext_mem_mod);
            tmod.addImport("cudaz_ext_sync", ext_sync_mod);
            tmod.addImport("ftw", ftw_mod); // lane-e E2: loadFromFtw test
            tmod.addImport("gguf", gguf_mod); // fixture GGUF del test
        }
        if (std.mem.eql(u8, tf, "src/matmul/hybrid_exec.zig")) {
            tmod.addImport("debug", debug_mod);
            tmod.addOptions("build_options", options);
            tmod.addImport("moe_hybrid", moe_hybrid_mod);
            tmod.addImport("moe_cache", moe_cache_mod);
            tmod.addImport("moe_cpu_executor", moe_cpu_executor_mod);
            tmod.addImport("matmul", matmul_mod);
            tmod.addImport("cudaz", cudaz_mod);
            tmod.addImport("time", time_mod);
        }
        if (std.mem.eql(u8, tf, "src/mem/budget.zig")) {
            tmod.addImport("debug", debug_mod); // lane-f Phase 3
        }
        if (std.mem.eql(u8, tf, "tests/test_tier_manager.zig")) {
            tmod.addImport("tier_manager", tier_manager_mod);
        }
        if (std.mem.eql(u8, tf, "src/bench/kld.zig")) {
            tmod.addImport("debug", debug_mod); // lane-b3 P3.5
        }
        if (std.mem.eql(u8, tf, "tests/test_presets.zig")) {
            const presets_test_mod = b.createModule(.{
                .root_source_file = b.path("src/cli/presets.zig"),
                .target = target,
                .optimize = optimize,
            });
            tmod.addImport("presets", presets_test_mod); // lane-b3 P3.4 (C5)
        }
        if (std.mem.eql(u8, tf, "src/engine_api/metrics.zig")) {
            tmod.addImport("time", time_mod); // lane-b3 T5
        }
        if (std.mem.eql(u8, tf, "src/bench/adaptive_bench.zig")) {
            // lane-b3 T6: el bench importa @import("adaptive_dm") que a
            // su vez requiere debug; resolvemos ambos a nivel test_files.
            const adaptive_dm_test_mod = b.createModule(.{
                .root_source_file = b.path("src/speculative/adaptive_dm.zig"),
                .target = target,
                .optimize = optimize,
            });
            adaptive_dm_test_mod.addImport("debug", debug_mod);
            tmod.addImport("debug", debug_mod);
            tmod.addImport("adaptive_dm", adaptive_dm_test_mod);
        }
        if (std.mem.eql(u8, tf, "src/bench/controller_overhead.zig")) {
            // lane-b3 T7: el overhead bench importa time + adaptive_dm.
            const adaptive_dm_test_mod = b.createModule(.{
                .root_source_file = b.path("src/speculative/adaptive_dm.zig"),
                .target = target,
                .optimize = optimize,
            });
            adaptive_dm_test_mod.addImport("debug", debug_mod);
            tmod.addImport("debug", debug_mod);
            tmod.addImport("time", time_mod);
            tmod.addImport("adaptive_dm", adaptive_dm_test_mod);
        }
        if (std.mem.eql(u8, tf, "src/loader/ftw.zig")) {
            tmod.addImport("gguf", gguf_mod); // lane-f Phase 3
            tmod.addImport("debug", debug_mod);
        }
        if (std.mem.eql(u8, tf, "src/cuda/graph_capture.zig")) {
            tmod.addImport("cudaz", cudaz_mod); // lane-f Phase 3
            tmod.addImport("debug", debug_mod);
        }
        tmod.addImport("norm", norm_mod);
        tmod.addImport("ffn", ffn_mod);
        tmod.addImport("rope", rope_mod);
        tmod.addImport("gqa", gqa_mod);
        tmod.addImport("embedding", embedding_mod);
        tmod.addImport("hybrid_attn", hybrid_attn_mod);
        tmod.addImport("hybrid_layer", hybrid_layer_mod);
        tmod.addImport("short_conv", short_conv_mod);
        tmod.addImport("layer_streamer", layer_streamer_mod);
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
        tmod.addImport("gguf_moe", gguf_moe_mod); // lane-e
        tmod.addImport("offload_cache", moe_offload_mod); // lane-e
        tmod.addImport("bandwidth", moe_bandwidth_mod); // lane-f 4.7: q* auto-split
        tmod.addImport("moe_cuda", moe_cuda_mod); // lane-e E3

        tmod.addImport("host_bank", host_bank_mod); // lane-e E8 (probe)
        tmod.addImport("moe_cpu_executor", moe_cpu_executor_mod); // alias lane-e E5-hybrid
        tmod.addImport("moe_layer", moe_layer_mod); // lane-e E5
        tmod.addImport("expert_streamer", moe_expert_stream_mod); // 4.10: ExpertStreamer
        tmod.addImport("debug", debug_mod);
        tmod.addImport("resources", resources_mod); // coordinador post-freeze
        tmod.addImport("speculative", speculative_mod);
        tmod.addImport("host_bank", host_bank_mod); // lane-d D2
        tmod.addImport("cudaz_ext_mem", cudaz_ext_mem_mod); // lane-d D2
        tmod.addImport("moe_cpu_gemv", moe_gemv_mod); // lane-f F1
        tmod.addImport("moe_cpu_executor", moe_cpu_executor_mod); // lane-f F2
        tmod.addImport("cpu_executor", moe_cpu_executor_mod); // alias lane-e E5-hybrid
        tmod.addImport("cudaz_ext_sync", ext_sync_mod); // lane-f F3
        tmod.addImport("vram_budget", vram_budget_mod); // lane-f F5
        tmod.addImport("auth", server_auth_mod); // lane-server-f2
        tmod.addImport("validation", server_validation_mod); // lane-server-f2
        tmod.addImport("audit", server_audit_mod); // lane-server-f2
        tmod.addImport("batching", server_batching_mod); // lane-server-f2
        tmod.addImport("token_sink", server_token_sink_mod); // lane-server-f2
        tmod.addImport("quite", server_quite_mod); // lane-server-f2
        tmod.addImport("httpx", httpx_mod); // lane-server-f2 T9
        tmod.addImport("server", server_mod); // lane-server-f2 T9
        tmod.addImport("inference", inference_mod); // lane-server-f2 T9
        const t = b.addTest(.{
            .root_module = tmod,
            .filters = test_filters, // coordinador: --test-filter (protocolo CPU lanes)
        });

        if (has_cuda) {
            tmod.linkSystemLibrary("cuda", .{});
            tmod.linkSystemLibrary("cudart", .{});
            if (cuda_lib_dir_exists) tmod.addLibraryPath(.{ .cwd_relative = cuda_lib_path });
            tmod.linkSystemLibrary("cublas", .{});
            for (dequant_objs) |obj| {
                tmod.addObjectFile(obj);
            }
            // lane-f Phase 3: launchers encode MXFP4/Q8_0 (test de paridad).
            if (quant_encode_obj) |obj| tmod.addObjectFile(obj);
        } else {
            // CI sin toolkit: ver comentario en exe_mod (línea ~1558).
            tmod.addCSourceFile(.{ .file = b.path("src/cuda/cuda_noop_stub.c"), .flags = &.{} });
        }
        const run_t = b.addRunArtifact(t);
        if (paged_cubin_install) |inst| {
            run_t.step.dependOn(&inst.step);
        }
        // El cubin de layer_kernels también debe estar instalado antes de
        // correr tests: loadModule() lee la ruta instalada (zig-out/lib).
        if (layer_cubin_install) |inst| {
            run_t.step.dependOn(&inst.step);
        }
        if (moe_cubin_install) |inst| {
            run_t.step.dependOn(&inst.step);
        }
        // lane-b1 (Dev A): el cubin KVarN debe estar instalado antes de
        // correr tests GPU (loadModule lee la ruta zig-out/lib).
        if (kvarn_cubin_install) |inst| {
            run_t.step.dependOn(&inst.step);
        }
        // lane-b1 (Dev A): cubin MMA smoke + decode-split.
        if (kvarn_mma_cubin_install) |inst| {
            run_t.step.dependOn(&inst.step);
        }
        if (kvarn_split_cubin_install) |inst| {
            run_t.step.dependOn(&inst.step);
        }
        // lane-b1 (Dev B): cubins FA portable/vec para tests GPU.
        if (fattn_kvarn_portable_install) |inst| {
            run_t.step.dependOn(&inst.step);
        }
        if (fattn_kvarn_vec_install) |inst| {
            run_t.step.dependOn(&inst.step);
        }
        // STUDY §5.2: el test de prefill chunked carga el cubin separado.
        if (prefill_cubin_install) |inst| {
            run_t.step.dependOn(&inst.step);
        }
        test_step.dependOn(&run_t.step);

        // Paso dedicado para iterar rápido sólo en el roundtrip KV q8_0:
        //   zig build test-kvq   (con DUMP_KVQUANT=1 activa breadcrumbs)
        if (std.mem.eql(u8, tf, "tests/test_kv_append_quant_gpu.zig")) {
            const kvq_step = b.step("test-kvq", "Run KV q8_0 append/decode/prefill GPU tests");
            kvq_step.dependOn(&run_t.step);
        }
        // Paso dedicado G1c (lane-c): paridad split-K + regresión de
        // cobertura seq>1024 (tokens_per_split dinámico).
        //   zig build test-pasplit  (flock .bench.lock)
        if (std.mem.eql(u8, tf, "tests/test_paged_attention_gpu.zig")) {
            const pasplit_step = b.step("test-pasplit", "Run paged attention GPU split-K parity + coverage tests");
            pasplit_step.dependOn(&run_t.step);
        }
        // F1v7 (lane-f, hallazgo 11.6): regresión del pairing RoPE —
        // llama-arch = NORM (pares consecutivos), no NEOX half-split.
        //   zig build test-rope-norm
        if (std.mem.eql(u8, tf, "tests/test_rope_norm.zig")) {
            const rope_norm_step = b.step("test-rope-norm", "Run RoPE NORM pairing regression test (lane-f 11.6)");
            rope_norm_step.dependOn(&run_t.step);
        }
        // KT-B (lane-f): runtime KV-transfer — gate-0 identity roundtrip +
        // formato .ktb + error paths.
        //   zig build test-ktb
        if (std.mem.eql(u8, tf, "tests/test_kt_transfer.zig")) {
            const ktb_step = b.step("test-ktb", "Run KT-B KV-transfer tests (lane-f, .ktb + gate-0 identity)");
            ktb_step.dependOn(&run_t.step);
        }
        // Presupuesto CPU central (coordinador, post-freeze 2026-09-13):
        //   zig build test-resources
        if (std.mem.eql(u8, tf, "tests/test_resources.zig")) {
            const res_step = b.step("test-resources", "Run central CPU/mem budget tests (coordinador post-freeze)");
            res_step.dependOn(&run_t.step);
        }
        // B-a1 (lane-a): paso dedicado — prefill/decode fused por formato
        //   zig build test-pafused   (paridad q4_k g8 + resto formatos + B-a3 + a-U3 q3_k dp4a)
        if (std.mem.eql(u8, tf, "tests/test_fused_decode_gpu.zig")) {
            const paf_step = b.step("test-pafused", "Run paged_attention fused prefill/decode parity tests (lane-a)");
            paf_step.dependOn(&run_t.step);
        }
        // lane-cuda UC-1.5: paso dedicado NVRTC — paridad JIT vs cubin del
        //   MISMO kernel + ErrorFlag smoke. Loop de iter kernel: edit del
        //   fuente embebido + `zig build test-nvrtc` SIN rebuild del binario.
        if (std.mem.eql(u8, tf, "tests/test_nvrtc_gpu.zig")) {
            const nvrtc_step = b.step("test-nvrtc", "Run NVRTC JIT parity tests vs build-time cubin (lane-cuda UC-1)");
            nvrtc_step.dependOn(&run_t.step);
        }
        // U2 (lane-b1): paso dedicado — prefill GPU dense batched paridad CPU.
        //   GGUF_MODEL_PATH=<híbrido> zig build test-u2
        if (std.mem.eql(u8, tf, "tests/test_u2_prefill_dense_gpu.zig")) {
            const u2_step = b.step("test-u2", "Run U2 prefill GPU dense batched parity tests (lane-b1)");
            u2_step.dependOn(&run_t.step);
        }
        // U2-kv-append (lane-b1): diagnóstico kvAppend batched vs unrolled.
        //   GGUF_MODEL_PATH=<híbrido> zig build test-u2-kv-append
        if (std.mem.eql(u8, tf, "tests/test_u2_kv_append_diag.zig")) {
            const u2ka_step = b.step("test-u2-kv-append", "Run U2 kvAppend batched vs unrolled diagnostic (lane-b1)");
            u2ka_step.dependOn(&run_t.step);
        }
        // 5.2 (lane-b1): draft-model DFlash — helpers + truncado p_min.
        //   zig build test-dflash
        if (std.mem.eql(u8, tf, "tests/test_dflash_draft.zig")) {
            const df_step = b.step("test-dflash", "Run DFlash draft-model tests (lane-b1 5.2)");
            df_step.dependOn(&run_t.step);
        }
        // 5.2 (lane-b1): integración con sidecar REAL (Qwen3.5-9B-DFlash).
        //   ZIG_AI_DFLASH_SIDECAR=<dflash.gguf> ZIG_AI_DFLASH_TARGET=<target.gguf> \
        //     zig build test-dflash-e2e
        if (std.mem.eql(u8, tf, "tests/test_dflash_e2e.zig")) {
            const dfe_step = b.step("test-dflash-e2e", "Run DFlash real-sidecar integration tests (lane-b1 5.2)");
            dfe_step.dependOn(&run_t.step);
        }
        // U2-llama (lane-b1): gate 512tok con capa real Llama (chasis U1).
        //   GGUF_MODEL_PATH=<llama-dense> zig build test-u2-llama
        if (std.mem.eql(u8, tf, "tests/test_u2_prefill_llama_gpu.zig")) {
            const u2l_step = b.step("test-u2-llama", "Run U2-llama 512tok real-layer parity tests (lane-b1)");
            u2l_step.dependOn(&run_t.step);
        }
        // U2-llama-stack (lane-b1): pila 28 capas + head, greedy top-1.
        //   GGUF_MODEL_PATH=<llama-dense> zig build test-u2-stack
        if (std.mem.eql(u8, tf, "tests/test_u2_llama_stack_gpu.zig")) {
            const u2s_step = b.step("test-u2-stack", "Run U2-llama full-stack 28-layer greedy parity tests (lane-b1)");
            u2s_step.dependOn(&run_t.step);
        }
        // Pasos dedicados lane-kvc: tracer K/V + harness predictor (P2).
        if (std.mem.eql(u8, tf, "tests/test_kv_trace.zig")) {
            const kvt_step = b.step("test-kv-trace", "Run KV tracer capture tests (lane-kvc)");
            kvt_step.dependOn(&run_t.step);
        }
        if (std.mem.eql(u8, tf, "tests/test_quant_weight_i2s.zig")) {
            const i2s_step = b.step("test-i2s", "Run BitNet i2_s QuantWeight dequant tests (lane-kvc)");
            i2s_step.dependOn(&run_t.step);
        }
        if (std.mem.eql(u8, tf, "tests/test_kv_predictor.zig")) {
            const kvp_step = b.step("test-kv-predictor", "Run KV predictor closed-loop GATE harness (lane-kvc)");
            kvp_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-f F1: iterar rápido sólo en GEMV CPU (sin GPU).
        if (std.mem.eql(u8, tf, "tests/test_cpu_gemv.zig")) {
            const gemv_step = b.step("test-gemv", "Run CPU GEMV q4_0/q8_0 parity tests + bench");
            gemv_step.dependOn(&run_t.step);
        }
        // Pasos dedicados lane-e (MOE_DEBUG=1 breadcrumbs; NOGATHER=1 A/B):
        if (std.mem.eql(u8, tf, "tests/test_moe_cache.zig")) {
            tmod.addImport("budget", mem_budget_mod); // lane-e E2+budget
            const moe_step = b.step("test-moe", "Run MoE parser/offload-cache tests");
            moe_step.dependOn(&run_t.step);
        }
        // lane-e 11.2: bundle test necesita el módulo expert_bundle
        if (std.mem.eql(u8, tf, "tests/test_expert_bundle.zig")) {
            tmod.addImport("expert_bundle", moe_bundle_mod); // lane-e 11.2
            const bundle_step = b.step("test-bundle", "Run MoE expert bundle integrity tests (lane-e 11.2)");
            bundle_step.dependOn(&run_t.step);
        }
        if (std.mem.eql(u8, tf, "tests/test_moe_bandwidth.zig")) {
            tmod.addImport("budget", mem_budget_mod); // paridad de imports con test_moe_cache
            const bw_step = b.step("test-bandwidth", "Run 4.7 q* auto-split tests");
            bw_step.dependOn(&run_t.step);
        }
        if (std.mem.eql(u8, tf, "tests/test_moe_e2e.zig")) {
            const moee2e_step = b.step("test-moe-e2e", "Run E2E hybrid MoE executor-merge test");
            moee2e_step.dependOn(&run_t.step);
        }
        if (std.mem.eql(u8, tf, "tests/test_moe_kernels_gpu.zig")) {
            const moegpu_step = b.step("test-moe-gpu", "Run MoE ensure_experts_moe GPU parity tests");
            moegpu_step.dependOn(&run_t.step);
        }
        if (std.mem.eql(u8, tf, "tests/test_moe_gather_gpu.zig")) {
            const moegather_step = b.step("test-moe-gather", "Run MoE gather fused GPU tests + GB/s bench");
            moegather_step.dependOn(&run_t.step);
        }
        if (std.mem.eql(u8, tf, "tests/test_moe_layer_gpu.zig")) {
            const moelayer_step = b.step("test-moe-layer", "Run MoE layer end-to-end GPU test");
            moelayer_step.dependOn(&run_t.step);
        }
        // Paso dedicado 4.10/4.12: streaming expert-por-experto + resize elástico.
        if (std.mem.eql(u8, tf, "tests/test_moe_expert_stream_gpu.zig")) {
            const moetstream_step = b.step("test-moe-stream", "Run MoE expert streaming 2-slot + elastic resize tests");
            moetstream_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-e (experimento MAP_SHARED, requiere GPU):
        if (std.mem.eql(u8, tf, "tests/test_moe_zerocopy_probe.zig")) {
            const zc_step = b.step("test-zerocopy-probe", "Probe HostBank register on RW vs RO mmap");
            zc_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-f F3: executor submit/sync (host path, sin GPU).
        if (std.mem.eql(u8, tf, "tests/test_hybrid_sync.zig")) {
            const hsync_step = b.step("test-hybridsync", "Run cpu_executor submit/sync handshake tests");
            hsync_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-f F5: aritmética rebuild elástico (sin GPU).
        if (std.mem.eql(u8, tf, "tests/test_vram_rebuild.zig")) {
            tmod.addImport("budget", mem_budget_mod); // lane-f P3: fitCheckElastic
            const vrb_step = b.step("test-vramrebuild", "Run vram_budget elastic rebuild tests");
            vrb_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-f Phase 3: paridad bit-exact encode MXFP4/Q8_0.
        if (std.mem.eql(u8, tf, "tests/test_quant_encode_gpu.zig")) {
            const qenc_step = b.step("test-quant-encode", "Run GPU encode MXFP4/Q8_0 parity tests");
            qenc_step.dependOn(&run_t.step);
        }
        // Paso dedicado STUDY §5.6: iterar en la paridad del ΔNet warp.
        if (std.mem.eql(u8, tf, "tests/test_deltanet_warp.zig")) {
            const dnw_step = b.step("test-dnwarp", "Run deltaNetWarp parity test");
            dnw_step.dependOn(&run_t.step);
        }
        // Paso dedicado STUDY §5.8: micro-bench MMQ vs qgemm en formas SSM.
        if (std.mem.eql(u8, tf, "tests/test_study_suite.zig")) {
            const s58_step = b.step("study-58", "Run study_suite (incluye §5.8 bench con STUDY_58_BENCH=1)");
            s58_step.dependOn(&run_t.step);
        }
        // Paso dedicado STUDY §5.2: paridad chunked ΔNet prefill.
        if (std.mem.eql(u8, tf, "tests/test_prefill_chunked.zig")) {
            const pc_step = b.step("test-prefillchunked", "Run prefill chunked parity test");
            pc_step.dependOn(&run_t.step);
        }
        // RLT training: gradient check test needs rlt_layer module
        if (std.mem.eql(u8, tf, "tests/test_rlt_backward.zig")) {
            tmod.addImport("rlt_layer", train_rlt_mod);
            const rlt_bwd_step = b.step("test-rlt-backward", "Run RLT gradient check (finite-diff vs analytic)");
            rlt_bwd_step.dependOn(&run_t.step);
        }
        if (std.mem.eql(u8, tf, "tests/test_rlt_train.zig")) {
            tmod.addImport("rlt_layer", train_rlt_mod);
            const rlt_train_step = b.step("test-rlt-train", "Run RLT training smoke test (50 steps)");
            rlt_train_step.dependOn(&run_t.step);
        }
        // RLT training: roundtrip export GGUF → parser real del engine
        if (std.mem.eql(u8, tf, "tests/test_rlt_export.zig")) {
            tmod.addImport("export_gguf", train_export_mod);
            const rlt_export_step = b.step("test-rlt-export", "Run RLT GGUF export roundtrip vs engine parser");
            rlt_export_step.dependOn(&run_t.step);
        }
        // RLT merge tool: sidecar→base byte-level
        if (std.mem.eql(u8, tf, "tests/test_rlt_merge.zig")) {
            tmod.addImport("export_gguf", train_export_mod);
            tmod.addImport("merge_tool", train_merge_mod);
            const rlt_merge_step = b.step("test-rlt-merge", "Run RLT merge tool roundtrip test");
            rlt_merge_step.dependOn(&run_t.step);
        }
        // R-4 (RLT training Fase 2): readRltcap roundtrip .rltcap v1
        if (std.mem.eql(u8, tf, "tests/test_rltcap.zig")) {
            tmod.addImport("train", train_mod);
            const rltcap_step = b.step("test-rltcap", "Run readRltcap roundtrip test (fixture /tmp/rltcap_test.rltcap)");
            rltcap_step.dependOn(&run_t.step);
        }
        // Paso dedicado STUDY §5.2 (1.4): paridad oráculo WY-chunk.
        if (std.mem.eql(u8, tf, "tests/test_prefill_wy.zig")) {
            const wy_step = b.step("test-prefillwy", "Run WY-chunk oracle parity test");
            wy_step.dependOn(&run_t.step);
        }
        // Paso dedicado STUDY §5.2 (1.4, lane-b): paridad kernels WY GPU.
        if (std.mem.eql(u8, tf, "tests/test_prefill_wy_gpu.zig")) {
            const wyg_step = b.step("test-prefillwygpu", "Run WY GPU kernel parity test (flock .bench.lock)");
            wyg_step.dependOn(&run_t.step);
        }
        if (std.mem.eql(u8, tf, "tests/test_dflash_encoder.zig")) {
            const dfe_step = b.step("test-dflash-enc", "Run DFlash encoder parity test (flock .bench.lock + env)");
            dfe_step.dependOn(&run_t.step);
        }
        // Paso dedicado 1.14 (lane-c): fórmula FLA GDN l2norm + tests inline ssm.
        if (std.mem.eql(u8, tf, "tests/test_gdn_l2norm.zig")) {
            tmod.addImport("ssm", ssm_mod); // 1.14 lane-c: tests inline de ssm.zig
            const gdn_step = b.step("test-gdn-l2norm", "Run GDN l2norm FLA formula tests (1.14; GPU part flock .bench.lock)");
            gdn_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-server-f2 T9: E2E server (SERVER_TEST_MODEL).
        if (std.mem.eql(u8, tf, "tests/test_server_e2e.zig")) {
            const e2e_step = b.step("test-server-e2e", "Run server E2E (needs SERVER_TEST_MODEL)");
            e2e_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-b2 (CPU only): KVarN roundtrip bit-exacto
        // + Hadamard 128 + Sinkhorn + K/V tile encode/decode.
        if (std.mem.eql(u8, tf, "tests/test_kvarn_roundtrip.zig")) {
            const kvarn_step = b.step("test-kvarn", "Run KVarN CPU reference roundtrip tests");
            kvarn_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-c (CPU only): 9.1 store KVarN en manager.
        if (std.mem.eql(u8, tf, "tests/test_kvarn_manager.zig")) {
            const kvarn_mgr_step = b.step("test-kvarn-manager", "Run KVarN manager integration tests (lane-c 9.1)");
            kvarn_mgr_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-b2 (CPU only): KVarN KLD/calidad vs BF16.
        if (std.mem.eql(u8, tf, "tests/test_kvarn_kld.zig")) {
            const kvarn_kld_step = b.step("test-kvarn-kld", "Run KVarN KLD/calidad tests (CPU reference)");
            kvarn_kld_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-b1 Dev-B: P2.2 capabilities + routing policy.
        if (std.mem.eql(u8, tf, "tests/test_backend_capabilities.zig")) {
            tmod.addImport("fattn_kvarn", fattn_kvarn_mod); // lane-b1 Dev-B parity test
            tmod.addImport("kvarn_kernels", kvarn_kernels_mod); // lane-b1 Dev-B parity test
            tmod.addOptions("build_options", kvarn_options); // lane-b1 Dev-B
            const bkcap_step = b.step("test-bkcap", "Run backend capabilities tests (lane-b1 Dev-B)");
            bkcap_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-b1 Dev-A: gate F0 WHT-128 device vs CPU B2.
        // Paso dedicado 9.5 (lane-b): ProfitController adaptive-dm — CPU-puro.
        if (std.mem.eql(u8, tf, "tests/test_adaptive_dm.zig")) {
            const adm_step = b.step("test-adaptive-dm", "Run 9.5 adaptive-dm ProfitController tests (CPU)");
            adm_step.dependOn(&run_t.step);
        }
        if (std.mem.eql(u8, tf, "tests/test_kvarn_wht_gpu.zig")) {
            tmod.addOptions("build_options", kvarn_options); // lane-b1 Dev-A
            const wht_step = b.step("test-kvarn-wht", "Run KVarN WHT-128 GPU smoke (gate F0)");
            wht_step.dependOn(&run_t.step);
        }
        // 9.12 (lane-cuda) F2/F3: D64 GPU smoke tests.
        if (std.mem.eql(u8, tf, "tests/test_kvarn_d64_gpu.zig")) {
            tmod.addOptions("build_options", kvarn_options);
            tmod.addImport("kvarn_kernels", kvarn_kernels_mod); // 9.12 F2: D64 store
            tmod.addImport("fattn_kvarn", fattn_kvarn_mod); // 9.12 F3: D64 fattn
            const d64_step = b.step("test-kvarn-d64", "Run KVarN D64 GPU smoke tests (WHT-64 + store + fattn)");
            d64_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-b1 Dev-A: A10 decode-split MMA E2E.
        if (std.mem.eql(u8, tf, "tests/test_kvarn_split_gpu.zig")) {
            tmod.addOptions("build_options", kvarn_options); // lane-b1 Dev-A
            tmod.addImport("kvarn_kernels", kvarn_kernels_mod); // lane-b1 Dev-A
            tmod.addImport("fattn_kvarn", fattn_kvarn_mod); // lane-b1 Dev-A
            const split_step = b.step("test-kvarn-split", "Run KVarN decode-split MMA E2E (A10)");
            split_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-b1 Dev-A: A9 MMA m16n8k16 smoke.
        if (std.mem.eql(u8, tf, "tests/test_kvarn_mma_gpu.zig")) {
            tmod.addOptions("build_options", kvarn_options); // lane-b1 Dev-A
            const mma_step = b.step("test-kvarn-mma", "Run KVarN MMA smoke (A9)");
            mma_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-b1 Dev-A: A8/M1 store GPU bit-exacto vs B2.
        if (std.mem.eql(u8, tf, "tests/test_kvarn_store_gpu.zig")) {
            tmod.addOptions("build_options", kvarn_options); // lane-b1 Dev-A
            tmod.addImport("kvarn_kernels", kvarn_kernels_mod); // lane-b1 Dev-A
            const store_step = b.step("test-kvarn-store", "Run KVarN store bit-exacto GPU tests (M1)");
            store_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-b1 Dev-B: B3 init_descs GPU.
        if (std.mem.eql(u8, tf, "tests/test_kvarn_init_descs_gpu.zig")) {
            tmod.addOptions("build_options", kvarn_options); // lane-b1 Dev-B
            tmod.addImport("kvarn_kernels", kvarn_kernels_mod); // lane-b1 Dev-B
            const b3_step = b.step("test-kvarn-initdescs", "Run KVarN init_descs GPU tests (B3)");
            b3_step.dependOn(&run_t.step);
        }
        // lane-b1 Dev-A M3: KvarnGpuCache surface (CPU validation).
        if (std.mem.eql(u8, tf, "tests/test_kvarn_gpu_cache.zig")) {
            tmod.addImport("kvarn_gpu_cache", kvarn_gpu_cache_mod); // lane-b1 Dev-A
            tmod.addImport("kvarn_kernels", kvarn_kernels_mod); // lane-b1 Dev-A
            tmod.addImport("fattn_kvarn", fattn_kvarn_mod); // lane-b1 Dev-A
            tmod.addImport("backend_capabilities", backend_capabilities_mod); // lane-b1 Dev-A
            tmod.addOptions("build_options", kvarn_options); // lane-b1 Dev-A
            const kgc_step = b.step("test-kvarn-gpucache", "Run KvarnGpuCache surface tests (M3)");
            kgc_step.dependOn(&run_t.step);
        }
        // lane-b1 Dev-A M3: patrón main.zig (prefill+decode × capas, probeDevice).
        if (std.mem.eql(u8, tf, "tests/test_kvarn_main_pattern_gpu.zig")) {
            tmod.addImport("kvarn_gpu_cache", kvarn_gpu_cache_mod); // lane-b1 Dev-A
            tmod.addImport("kvarn_kernels", kvarn_kernels_mod); // lane-b1 Dev-A
            tmod.addImport("backend_capabilities", backend_capabilities_mod); // lane-b1 Dev-A
            tmod.addOptions("build_options", kvarn_options); // lane-b1 Dev-A
            const mp_step = b.step("test-kvarn-mainpattern", "Run KVarN main.zig-pattern integration (M3)");
            mp_step.dependOn(&run_t.step);
        }
        // lane-b1 Dev-A: bisect del hang store kvh=2 (layout real).
        if (std.mem.eql(u8, tf, "tests/test_kvarn_iso_store_gpu.zig")) {
            tmod.addImport("kvarn_kernels", kvarn_kernels_mod); // lane-b1 Dev-A
            tmod.addImport("kv_cache", kv_cache_mod); // lane-b1 Dev-A
            tmod.addOptions("build_options", kvarn_options); // lane-b1 Dev-A
            const iso_step = b.step("test-kvarn-isostore", "Run KVarN iso-store kvh=2 bisect (lane-b1)");
            iso_step.dependOn(&run_t.step);
        }
        // Pasos dedicados lane-b1 Dev-B: FA portable/vec/bench GPU.
        if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_gpu.zig")) {
            tmod.addOptions("build_options", kvarn_options); // lane-b1 Dev-B
            tmod.addImport("fattn_kvarn", fattn_kvarn_mod); // lane-b1 Dev-B
            tmod.addImport("kvarn_kernels", kvarn_kernels_mod); // lane-b1 Dev-B
            const b5_step = b.step("test-fattn-kvarn", "Run KVarN portable FA GPU tests (B5)");
            b5_step.dependOn(&run_t.step);
        }
        if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_wht_gpu.zig")) {
            tmod.addOptions("build_options", kvarn_options); // lane-b1 Dev-B
            const b4wht_step = b.step("test-fattn-kvarn-wht", "Run KVarN FA WHT prep tests (B4/D3)");
            b4wht_step.dependOn(&run_t.step);
        }
        if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_vec_gpu.zig")) {
            tmod.addOptions("build_options", kvarn_options); // lane-b1 Dev-B
            tmod.addImport("fattn_kvarn", fattn_kvarn_mod); // lane-b1 Dev-B
            tmod.addImport("kvarn_kernels", kvarn_kernels_mod); // lane-b1 Dev-B
            const b6_step = b.step("test-fattn-kvarn-vec", "Run KVarN vec FA GPU tests (B6)");
            b6_step.dependOn(&run_t.step);
        }
        if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_bench_gpu.zig")) {
            tmod.addOptions("build_options", kvarn_options); // lane-b1 Dev-B
            tmod.addImport("fattn_kvarn", fattn_kvarn_mod); // lane-b1 Dev-B
            tmod.addImport("backend_capabilities", backend_capabilities_mod); // lane-b1 Dev-B
            tmod.addImport("kvarn_kernels", kvarn_kernels_mod); // lane-b1 Dev-B (iter-18)
            const b8_step = b.step("test-fattn-kvarn-bench", "Run KVarN FA bench skeleton tests (B8)");
            b8_step.dependOn(&run_t.step);
        }
        // Pasos lane-b1 Dev-B iter8: M1-seed / D256 / D4-tail (mismos imports).
        if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_m1_seed_gpu.zig") or std.mem.eql(u8, tf, "tests/test_fattn_kvarn_d256_gpu.zig") or std.mem.eql(u8, tf, "tests/test_fattn_kvarn_d512_gpu.zig") or std.mem.eql(u8, tf, "tests/test_fattn_kvarn_multistream_gpu.zig") or std.mem.eql(u8, tf, "tests/test_fattn_kvarn_d4_tail_gpu.zig") or std.mem.eql(u8, tf, "tests/test_fattn_kvarn_smoke_gpu.zig") or std.mem.eql(u8, tf, "tests/test_fattn_kvarn_repro_gpu.zig")) { // lane-b1
            tmod.addOptions("build_options", kvarn_options); // lane-b1 Dev-B
            tmod.addImport("fattn_kvarn", fattn_kvarn_mod); // lane-b1 Dev-B
            tmod.addImport("kvarn_kernels", kvarn_kernels_mod); // lane-b1 Dev-B
            const name = if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_m1_seed_gpu.zig"))
                "test-fattn-kvarn-m1seed"
            else if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_d256_gpu.zig"))
                "test-fattn-kvarn-d256"
            else if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_repro_gpu.zig"))
                "test-fattn-kvarn-repro"
            else if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_d512_gpu.zig"))
                "test-fattn-kvarn-d512"
            else if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_multistream_gpu.zig"))
                "test-fattn-kvarn-multistream"
            else if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_d4_tail_gpu.zig"))
                "test-fattn-kvarn-d4tail"
            else
                "test-fattn-kvarn-smoke"; // lane-b1 Dev-B iter-28
            const desc = if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_m1_seed_gpu.zig"))
                "Run KVarN FA M1 1000-seed tests"
            else if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_d256_gpu.zig"))
                "Run KVarN FA D256 E2E tests"
            else if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_repro_gpu.zig"))
                "Run KVarN FA E2E repro (real dumps)"
            else if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_d512_gpu.zig"))
                "Run KVarN FA D512 E2E tests"
            else if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_multistream_gpu.zig"))
                "Run KVarN FA multistream matrix tests"
            else if (std.mem.eql(u8, tf, "tests/test_fattn_kvarn_d4_tail_gpu.zig"))
                "Run KVarN D4 tail materialize tests"
            else
                "Run KVarN FA ARITY smoke self-tests"; // lane-b1 Dev-B iter-28
            const it_step = b.step(name, desc);
            it_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-b2 (CPU): KVCPT integración kv_cache_manager.
        if (std.mem.eql(u8, tf, "tests/test_kvcpt_integration.zig")) {
            const kvcpt_step = b.step("test-kvcpt-integration", "Run KVCPT integration tests");
            kvcpt_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-b2 (CPU only): P0.3 cache types q2_0s/q2_1/q3_0/q3_1/q6_0/q6_1.
        if (std.mem.eql(u8, tf, "tests/test_kv_p03_types.zig")) {
            const p03_step = b.step("test-kv-p03", "Run P0.3 cache types roundtrip (CPU)");
            p03_step.dependOn(&run_t.step);
        }
        // Paso dedicado lane-b2 (CPU only): P0.2 KVCPT cola exacta + rollback.
        if (std.mem.eql(u8, tf, "tests/test_kv_tail.zig")) {
            const tail_step = b.step("test-kv-tail", "Run KVCPT precision-tail tests (CPU)");
            tail_step.dependOn(&run_t.step);
        }
        // Paso dedicado mmproj (CPU only): unit tests de vision.
        if (std.mem.eql(u8, tf, "tests/test_mmproj.zig")) {
            tmod.addImport("mrope_vision", mrope_vision_mod); // mmproj
            tmod.addImport("conv2d", vision_conv2d_mod); // mmproj
            tmod.addImport("preprocess", vision_preprocess_mod); // mmproj
            tmod.addImport("token_inject", vision_token_inject_mod); // mmproj
            tmod.addImport("mmproj_config", mmproj_config_mod); // mmproj
            tmod.addImport("mmproj_model", mmproj_model_mod); // mmproj
            tmod.addImport("vision_clip_encoder", vision_clip_encoder_mod); // mmproj
            tmod.addImport("vision_clip_gpu", clip_gpu_mod); // mmproj 10.2-fix: paridad CPU/GPU
            tmod.addImport("vision_video", vision_video_mod); // mmproj 10.7: pipeline video
            tmod.addImport("clip_block", clip_block_mod); // mmproj
            const mmproj_step = b.step("test-mmproj", "Run mmproj/vision unit tests (PLAN_MMPROJ)");
            mmproj_step.dependOn(&run_t.step);
        }
    }

    // P0-6 dp4a parity debug test: edit tests/test_p0_6_dp4a.zig
    // Note: file-exists check omitted — zig 0.16 Dir requires Io for access.
    // If file is absent, b.path() creates a lazy path that fails clearly at build time.
    {
        const tmod = b.createModule(.{
            .root_source_file = b.path("tests/test_p0_6_dp4a.zig"),
            .target = target,
            .optimize = optimize,
        });
        tmod.link_libc = true;
        tmod.addImport("core", core_mod);
        tmod.addImport("matmul", matmul_mod);
        tmod.addImport("fa", fa_mod);
        tmod.addImport("transformer", transformer_mod);
        tmod.addImport("kv_cache", kv_cache_mod);
        tmod.addImport("gguf", gguf_mod); // lane-kvc P1: dequant i2_s test fixture
        tmod.addImport("cudaz", cudaz_mod);
        tmod.addImport("cudaz_ext_mem", cudaz_ext_mem_mod);
        tmod.addImport("debug", debug_mod);
        tmod.addImport("gguf", gguf_mod);
        tmod.addImport("model_config", model_config_mod); // lane-kvc P4: repro bitnet load
        tmod.addImport("gguf_model", gguf_model_mod); // lane-kvc P4: idem (emb+lm_head)
        tmod.addImport("gguf_moe", gguf_moe_mod);
        tmod.addImport("cuda_runtime", cuda_runtime_mod);
        tmod.addImport("embedding", embedding_mod); // lane-b1 7.1d: paridad lookup cuant-residente
        tmod.addImport("quant_weight", quant_weight_mod); // lane-b1 7.1d
        tmod.addImport("short_conv", short_conv_mod); // lane-f 8.3: paridad shortconv scratch
        tmod.addImport("layer_kernels", layer_kernels_mod); // lane-f 8.3: idem (forwardGPU)
        tmod.addImport("tensor", core_mod); // lane-f 8.3: Tensor(f32) wrapper (core = tensor.zig)
        tmod.addImport("cublas", cublas_mod); // lane-f 8.3: GpuTensor para el readback
        tmod.addImport("time", time_mod); // lane-f 1.3: Timer.now() del bench scratch
        tmod.addImport("paged_attention", paged_attention_mod); // lane-f F3: repro iq2_s
        tmod.addImport("moe_layer", moe_layer_mod); // lane-f F/C: chain MoE scratch
        tmod.addImport("moe_cuda", moe_cuda_mod); // lane-f F/C
        tmod.addImport("offload_cache", moe_offload_mod); // lane-f F/C
        tmod.addImport("host_bank", host_bank_mod); // lane-f F/C: copy-once
        tmod.addImport("model_config", model_config_mod); // lane-f F/C: dump config scratch
        tmod.addImport("mmproj_model", mmproj_model_mod); // lane-mmproj 10.4: paridad encoder scratch
        tmod.addImport("gguf_tokenizer", gguf_tokenizer_mod); // lane-f F1: repro PPL multi-token
        tmod.addImport("bpe", tokenizer_mod); // lane-f F1: idem
        tmod.addImport("pipeline", pipeline_mod); // lane-f F1: InferencePipeline repro
        tmod.addImport("norm", norm_mod); // lane-f F1: rmsNorm final del incremental
        tmod.addImport("speculative", speculative_mod); // lane-c 5.1: paridad DflashEncoder
        tmod.addImport("mmproj_config", mmproj_config_mod); // lane-mmproj
        tmod.addImport("vision_clip_encoder", vision_clip_encoder_mod); // lane-mmproj
        tmod.addImport("vision_clip_gpu", clip_gpu_mod); // lane-mmproj
        tmod.addImport("vision_preprocess", vision_preprocess_mod); // lane-mmproj
        tmod.addImport("vision_conv2d", vision_conv2d_mod); // lane-mmproj 10.2-bisect
        tmod.addImport("mrope_vision", mrope_vision_mod); // lane-mmproj
        tmod.addImport("clip_block", clip_block_mod); // lane-mmproj
        tmod.addImport("kvarn_kernels", kvarn_kernels_mod); // lane-b 9.4: store D256 debug scratch
        tmod.addImport("nvrtc", nvrtc_mod); // lane-cuda: A/B dp4a JIT scratch
        tmod.addOptions("build_options", kvarn_options); // lane-b 9.4: kvarn_cubin path
        if (has_cuda) { // lane-f 8.3: paridad GPU scratch necesita runtime CUDA
            tmod.linkSystemLibrary("cuda", .{});
            tmod.linkSystemLibrary("cudart", .{});
            if (cuda_lib_dir_exists) tmod.addLibraryPath(.{ .cwd_relative = cuda_lib_path });
            tmod.linkSystemLibrary("cublas", .{});
            // lane-f F1: los módulos importados (gguf/quant_weight) declaran
            // extern los launchers dequant de kernels/*.cu — enlazar los
            // objetos (mismo patrón que exe_mod/tests del trunk).
            for (dequant_objs) |obj| {
                tmod.addObjectFile(obj);
            }
        } else {
            tmod.addCSourceFile(.{ .file = b.path("src/cuda/cuda_noop_stub.c"), .flags = &.{} });
        }
        const dbg_test = b.addTest(.{
            .root_module = tmod,
            .filters = test_filters, // coordinador: --test-filter
        });
        const run_t = b.addRunArtifact(dbg_test);
        const dp4a_step = b.step("test-dp4a-p06", "Run P0-6 dp4a parity test (edit tests/test_p0_6_dp4a.zig)");
        dp4a_step.dependOn(&run_t.step);
    }
    // (lane-mmproj 10.7 moved ITS scratch tests to tests/test_mmproj.zig as
    // formal regressions — dbg step repurposed: AGENTS.md documents it as UX.)

    // === Scratch test genérico para iteración rápida INFRA ===
    // Edita `tests/test_dbg_scratch.zig` y ejecuta `zig build dbg` para
    // compilar/ejecutar solo este test sin tocar la suite completa.
    {
        const scratch_mod = b.createModule(.{
            .root_source_file = b.path("tests/test_dbg_scratch.zig"),
            .target = target,
            .optimize = optimize,
        });
        scratch_mod.link_libc = true;
        scratch_mod.addImport("core", core_mod);
        scratch_mod.addImport("debug", debug_mod);
        const scratch_test = b.addTest(.{
            .root_module = scratch_mod,
            .filters = test_filters,
        });
        const run_scratch = b.addRunArtifact(scratch_test);
        const dbg_step = b.step("dbg", "Run scratch test (edit tests/test_dbg_scratch.zig)");
        dbg_step.dependOn(&run_scratch.step);
    }

    // === lane-e: moe-bench (FFN MoE offload) + moe-make-fixture ===
    const mb_mod = b.createModule(.{
        .root_source_file = b.path("examples/moe_bench.zig"),
        .target = target,
        .optimize = optimize,
    });
    mb_mod.addImport("gguf", gguf_mod);
    mb_mod.addImport("debug", debug_mod);
    mb_mod.addImport("cudaz", cudaz_mod);
    mb_mod.addImport("layer_kernels", layer_kernels_mod);
    mb_mod.addImport("moe_cuda", moe_cuda_mod);
    mb_mod.addImport("offload_cache", moe_offload_mod);
    mb_mod.addImport("gguf_moe", gguf_moe_mod);
    mb_mod.addImport("moe_layer", moe_layer_mod);
    mb_mod.addImport("host_bank", host_bank_mod);
    mb_mod.addImport("moe_cpu_gemv", moe_gemv_mod);
    mb_mod.addImport("moe_cpu_executor", moe_cpu_executor_mod);
    mb_mod.addImport("expert_bundle", moe_bundle_mod); // lane-e 11.2: bundle contiguo
    mb_mod.link_libc = true;
    if (has_cuda) {
        mb_mod.linkSystemLibrary("cuda", .{});
        if (cuda_lib_dir_exists) mb_mod.addLibraryPath(.{ .cwd_relative = cuda_lib_path });
    } else {
        mb_mod.addCSourceFile(.{
            .file = b.path("src/cuda/cuda_noop_stub.c"),
            .flags = &.{},
        });
    }
    const mb = b.addExecutable(.{ .name = "moe-bench", .root_module = mb_mod });
    b.installArtifact(mb);
    if (layer_cubin_install) |inst| b.getInstallStep().dependOn(&inst.step);
    if (moe_cubin_install) |inst| b.getInstallStep().dependOn(&inst.step);
    const run_mb = b.addRunArtifact(mb);
    if (b.args) |args| run_mb.addArgs(args);
    const mb_step = b.step("moe-bench", "Run MoE offload FFN benchmark");
    mb_step.dependOn(&run_mb.step);

    const mf_mod = b.createModule(.{
        .root_source_file = b.path("examples/moe_make_fixture.zig"),
        .target = target,
        .optimize = optimize,
    });
    const mf = b.addExecutable(.{ .name = "moe-make-fixture", .root_module = mf_mod });
    b.installArtifact(mf);
    const run_mf = b.addRunArtifact(mf);
    if (b.args) |args| run_mf.addArgs(args);
    const mf_step = b.step("moe-make-fixture", "Generate synthetic MoE GGUF fixture");
    mf_step.dependOn(&run_mf.step);

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
    bench_pa_mod.addImport("debug", debug_mod);

    bench_pa_mod.link_libc = true;

    if (has_cuda) {
        bench_pa_mod.linkSystemLibrary("cuda", .{});
        bench_pa_mod.linkSystemLibrary("cudart", .{});
        if (cuda_lib_dir_exists) bench_pa_mod.addLibraryPath(.{ .cwd_relative = cuda_lib_path });
        bench_pa_mod.linkSystemLibrary("cublas", .{});
    } else {
        bench_pa_mod.addCSourceFile(.{
            .file = b.path("src/cuda/cuda_noop_stub.c"),
            .flags = &.{},
        });
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
    // === Lane A: bench_kv_quant (KV cuantizado memoria/tok-s) ===
    const bench_kv_quant_step = b.step("bench-kv-quant", "Lane A: KV quantized memory/throughput benchmark");
    const bench_kv_quant = b.addExecutable(.{
        .name = "bench_kv_quant",
        .root_module = bench_kv_quant_mod,
    });
    b.installArtifact(bench_kv_quant);
    const run_bench_kv_quant = b.addRunArtifact(bench_kv_quant);
    if (b.args) |args| run_bench_kv_quant.addArgs(args);
    bench_kv_quant_step.dependOn(&run_bench_kv_quant.step);

    // === RLT merge micro-benchmark (CPU only, no CUDA needed) ===
    const bench_rlt_step = b.step("bench-rlt", "RLT merge feedback micro-benchmark (CPU)");
    const bench_rlt_mod = b.createModule(.{
        .root_source_file = b.path("benchmarks/bench_rlt_merge.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    const bench_rlt = b.addExecutable(.{
        .name = "bench_rlt_merge",
        .root_module = bench_rlt_mod,
    });
    b.installArtifact(bench_rlt);
    const run_bench_rlt = b.addRunArtifact(bench_rlt);
    bench_rlt_step.dependOn(&run_bench_rlt.step);

    // === RLT Training CLI ===
    const train_cli_step = b.step("train", "Train RLT adapter (synthetic hidden states)");
    const train_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/train/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    train_cli_mod.addImport("rlt_layer", train_rlt_mod);
    train_cli_mod.addImport("export_gguf", train_export_mod);
    train_cli_mod.addImport("train", train_mod);
    train_cli_mod.link_libc = true;
    const train_cli = b.addExecutable(.{
        .name = "zig-ai-train",
        .root_module = train_cli_mod,
    });
    b.installArtifact(train_cli);
    const run_train_cli = b.addRunArtifact(train_cli);
    if (b.args) |args| run_train_cli.addArgs(args);
    train_cli_step.dependOn(&run_train_cli.step);

    // === RLT Merge CLI: inyectar sidecar entrenado en GGUF base ===
    const rlt_merge_cli_step = b.step("rlt-merge", "Merge RLT sidecar GGUF into base model (byte-level)");
    const rlt_merge_cli_mod = b.createModule(.{
        .root_source_file = b.path("src/train/merge_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    rlt_merge_cli_mod.addImport("merge_tool", train_merge_mod);
    rlt_merge_cli_mod.link_libc = true;
    const rlt_merge_cli = b.addExecutable(.{
        .name = "zig-ai-rlt-merge",
        .root_module = rlt_merge_cli_mod,
    });
    b.installArtifact(rlt_merge_cli);
    const run_rlt_merge_cli = b.addRunArtifact(rlt_merge_cli);
    if (b.args) |args| run_rlt_merge_cli.addArgs(args);
    rlt_merge_cli_step.dependOn(&run_rlt_merge_cli.step);

    // === Lane D: stream-bench (harness streaming denso, D4/D5) ===
    const stream_bench_step = b.step("stream-bench", "Lane D: streaming FFN benchmark (wire vs f32)");
    stream_bench_mod.link_libc = true;
    if (has_cuda) {
        stream_bench_mod.linkSystemLibrary("cuda", .{});
        stream_bench_mod.linkSystemLibrary("cudart", .{});
        stream_bench_mod.linkSystemLibrary("cublas", .{});
        if (cuda_lib_dir_exists) stream_bench_mod.addLibraryPath(.{ .cwd_relative = cuda_lib_path });
    } else {
        stream_bench_mod.addCSourceFile(.{
            .file = b.path("src/cuda/cuda_noop_stub.c"),
            .flags = &.{},
        });
    }
    const stream_bench = b.addExecutable(.{ .name = "stream-bench", .root_module = stream_bench_mod });
    b.installArtifact(stream_bench);
    const run_stream_bench = b.addRunArtifact(stream_bench);
    if (b.args) |args| run_stream_bench.addArgs(args);
    stream_bench_step.dependOn(&run_stream_bench.step);

    // === Lane-B3: bench-adaptive (micro-bench ProfitController off vs profit) ===
    // El módulo bench_adaptive_lib es la lógica reutilizable
    // (src/bench/adaptive_bench.zig). El ejecutable "bench-adaptive"
    // tiene como root examples/bench_adaptive.zig (driver standalone con
    // main) e importa la lógica vía bench_adaptive_lib.
    const bench_adaptive_lib = b.createModule(.{
        .root_source_file = b.path("src/bench/adaptive_bench.zig"),
        .target = target,
        .optimize = optimize,
    });
    const adaptive_dm_mod = b.createModule(.{
        .root_source_file = b.path("src/speculative/adaptive_dm.zig"),
        .target = target,
        .optimize = optimize,
    });
    adaptive_dm_mod.addImport("debug", debug_mod); // DUMP_SPEC/PERF_SPEC
    bench_adaptive_lib.addImport("debug", debug_mod);
    bench_adaptive_lib.addImport("adaptive_dm", adaptive_dm_mod);

    const bench_adaptive_exe_mod = b.createModule(.{
        .root_source_file = b.path("examples/bench_adaptive.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench_adaptive_exe_mod.addImport("debug", debug_mod);
    bench_adaptive_exe_mod.addImport("adaptive_dm", adaptive_dm_mod);
    bench_adaptive_exe_mod.addImport("adaptive_bench", bench_adaptive_lib);

    const bench_adaptive_exe = b.addExecutable(.{
        .name = "bench-adaptive",
        .root_module = bench_adaptive_exe_mod,
    });
    b.installArtifact(bench_adaptive_exe);
    const bench_adaptive_step = b.step("bench-adaptive", "Lane-B3: ProfitController off vs profit (micro-bench CPU, sin modelo)");
    const run_bench_adaptive = b.addRunArtifact(bench_adaptive_exe);
    if (b.args) |args| run_bench_adaptive.addArgs(args);
    bench_adaptive_step.dependOn(&run_bench_adaptive.step);

    // === Lane-B3: bench-overhead (coste CPU del controller por ciclo) ===
    // Mide ns/op del ciclo recordAccepted+recordTiming+evaluateProbe+tick.
    // Útil para el gate < 0.5% decode del LANE prompt: si el controller
    // cuesta < 1µs/op y el decode cuesta > 200µs/tok, el overhead es
    // 0.5% (regla Bee). El módulo bench_overhead_lib re-exporta la
    // lógica (src/bench/controller_overhead.zig); el binario "bench-overhead"
    // tiene como root examples/bench_overhead.zig.
    const bench_overhead_lib = b.createModule(.{
        .root_source_file = b.path("src/bench/controller_overhead.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench_overhead_lib.addImport("debug", debug_mod);
    bench_overhead_lib.addImport("time", time_mod);
    bench_overhead_lib.addImport("adaptive_dm", adaptive_dm_mod);

    const bench_overhead_exe_mod = b.createModule(.{
        .root_source_file = b.path("examples/bench_overhead.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_overhead_exe_mod.addImport("debug", debug_mod);
    bench_overhead_exe_mod.addImport("time", time_mod);
    bench_overhead_exe_mod.addImport("adaptive_dm", adaptive_dm_mod);
    bench_overhead_exe_mod.addImport("controller_overhead", bench_overhead_lib);

    const bench_overhead_exe = b.addExecutable(.{
        .name = "bench-overhead",
        .root_module = bench_overhead_exe_mod,
    });
    b.installArtifact(bench_overhead_exe);
    const bench_overhead_step = b.step("bench-overhead", "Lane-B3: ns/op del ciclo del ProfitController (micro-bench CPU)");
    const run_bench_overhead = b.addRunArtifact(bench_overhead_exe);
    if (b.args) |args| run_bench_overhead.addArgs(args);
    bench_overhead_step.dependOn(&run_bench_overhead.step);
}

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
