//! Lanza los kernels elementwise de `layer_kernels.cu` vía la CUDA driver API.
//! Los kernels corren en el stream compartido para ordenarse con las GEMM de
//! cuBLAS (mismo stream); se sincroniza una sola vez por token desde el llamador.
const std = @import("std");
const cudaz = @import("cudaz");
const build_options = @import("build_options");
const debugz = @import("debug");
const nvrtc = @import("nvrtc"); // lane-cuda UC-1.3: JIT opt-in con fallback cubin
const bridge = @import("bridge"); // lane-cuda UC-3.2: getT() type-safe

var g_module: ?cudaz.CUmodule = null;

/// Control global de la ruta de pesos cuantizados (Q4_0 GEMM device).
/// Se ajusta con `--quant`; los env `NOQ4*` siguen siendo un override.
pub var quant_enabled: bool = true;

pub fn quantPath() bool {
    return quant_enabled and !debugz.dbg.no_q4;
}

fn loadModule() !cudaz.CUmodule {
    if (g_module) |m| return m;
    const cubin_path = build_options.layer_cubin;
    if (cubin_path.len == 0) return error.CudaUnavailable;
    try cudaz.ensureContext();
    // UC-1.3 (lane-cuda): gate ZIG_AI_NVRTC=1 ⇒ compila layer_kernels.cu por
    // JIT (NVRTC, ~segundos) en vez del cubin build-time. Fallback
    // transparente: cualquier fallo del camino JIT cae al cubin original.
    // Trampa UC-1.4: el .cu usa headers del SDK (mma.h/cuda_fp16.h) que
    // NVRTC no trae — se inyectan via ZIG_AI_NVRTC_INC (default /usr/include).
    if (nvrtc.tryJitOrFallback(std.heap.c_allocator, .{
        .name = "layer_kernels",
        .src_path = nvrtc.srcPath(),
        .include_dirs = nvrtc.sdkIncludeDirs(),
    }, cubin_path)) |m| {
        g_module = m;
        if (debugz.dbg.dump_graph) dumpFuncs();
        return m;
    } else |e| {
        if (debugz.dbg.at(.info) and e != error.NvrtcDisabled) {
            debugz.dbg.printLevel(.info, "[nvrtc] JIT de layer_kernels falló ({s}) — usando cubin build-time\n", .{@errorName(e)});
        }
    }
    g_module = try cudaz.cuModuleLoad(cubin_path);
    if (debugz.dbg.dump_graph) dumpFuncs();
    return g_module.?;
}

/// STUDY §5.2: el prefill chunked vive en su PROPIO cubin (nvcc -cubin sólo
/// admite un input con -o). Lo cargamos como segundo módulo global.
/// Público para que los tests (test_prefill_chunked) lancen el kernel directo.
var g_prefill_module: ?cudaz.CUmodule = null;
pub fn loadPrefillModule() !cudaz.CUmodule {
    if (g_prefill_module) |m| return m;
    const cubin_path = build_options.prefill_cubin;
    if (cubin_path.len == 0) return error.CudaUnavailable;
    try cudaz.ensureContext();
    g_prefill_module = try cudaz.cuModuleLoad(cubin_path);
    return g_prefill_module.?;
}

/// Breadcrumb DUMP_GRAPH: imprime la dirección de cada kernel del cubin para
/// poder identificar qué nodo del grafo es cada función. SIEMPRE presente.
fn dumpFuncs() void {
    const mod = g_module.?;
    for (kernel_names) |kn| {
        const f = cudaz.cuModuleGetFunction(mod, kn) catch continue;
        debugz.dbg.print("DUMP_GRAPH func {x} = {s}\n", .{ @intFromPtr(f), kn });
    }
}

const kernel_names = [_][:0]const u8{
    "mulKernel", // 8.3: mul element-wise (gating GDN ShortConv LFM2)
    "addInplaceKernel",
    "addKernel",
    "conv1dSiluKernel",
    "conv1dLinearKernel", // 8.3: conv lineal SIN silu (LFM2 ssm_conv)
    "copyF16toF32Kernel",
    "copyF32toF16Kernel",
    "dflashTapsGatherKernel", // C6.1 (5.1, lane-c): gather taps dflash sin D2H
    "dflashFcGemmM1Kernel", // C6.1 (5.1, lane-c): GEMV q8_0 fc split-K (K=32k > smem M1 genérico)
    "deltaNetKernel",
    "deltaNetWarpKernel", // STUDY §5.6: warp-shuffle sin barreras
    "deltaNetFusedKernel", // STUDY §5.1: l2+ΔNet(registros)+rmsNorm — 3→1
    "q4gemmM1Dp4aKernel", // STUDY §5.8: GEMV q4_0 M=1 dp4a+quantizeA fused
    "q4gemmMDp4aKernel", // STUDY §5.9: GEMM q4_0 M≤32 dp4a — prefill tier
    "q5gemmM1Kernel", // STUDY 1.1: GEMV q5_k M=1 (ssm_out)
    "q6gemmM1Kernel", // STUDY 1.2: GEMV q6_k M=1 (lm_head/FFN-down)
    "q4kGemmM1Kernel", // U3 (eje §12): GEMV q4_k M=1 — hueco del Q4_K_XL/gate 3B
    "q8kGemmM1Kernel", // U3 (eje §12): GEMV q8_0 M=1 — 2º hueco (ssm_* ΔNet)
    "q3kGemmM1Dp4aKernel", // a-U3 (eje §12): GEMV q3_k M=1 dp4a — cuello 3B Q3_K_S
    "q3kGemmM1Dp4aPackedKernel", // a-U3 fase 3: ídem sobre repack 128B/SB coalesced
    "q3kGemmM1Dp4aPackedQKVKernel", // a-U3 fase 3c: fusión q/k/v — 1 launch, fase-1 única
    "q41GemmM1Dp4aKernel",    // P0-6: q4_1 M=1 dp4a
    "q2kGemmM1Dp4aKernel",    // P0-6: q2_k M=1 dp4a
    "iq3sGemmM1Dp4aKernel",   // P0-6: iq3_s M=1 dp4a
    "iq2sGemmM1Dp4aKernel",   // P0-6: iq2_s M=1 dp4a
    "iq4xsGemmM1Dp4aKernel",  // P0-6: iq4_xs M=1 dp4a
    "iq4nlGemmM1Dp4aKernel",  // P0-6: iq4_nl M=1 dp4a
    "iq3xxsGemmM1Dp4aKernel", // P0-6: iq3_xxs M=1 dp4a
    "iq2xxsGemmM1Dp4aKernel", // P0-6: iq2_xxs M=1 dp4a
    "iq2xsGemmM1Dp4aKernel",  // P0-6: iq2_xs M=1 dp4a
    "prefillDeltaNetChunk_nkda_K64", // STUDY §5.2: chunked batched ΔNet prefill
    "prefillDeltaNetChunk_nkda_K128",
    "prefillDeltaNetChunk_kda_K64",
    "prefillDeltaNetChunk_kda_K128",
    "embeddingGatherKernel",
    "gateComputeKernel",
    "gateKernel",
    "kvAppendF16Kernel",
    "kvAppendQ4_0Kernel",
    "dflash2TopKKernel",
    "dflash2TreeWalkKernel",
    "mmqQuantizeAQ8Kernel",
    "mmqQ4_0GEMVKernel",
    "mmqQ4_0GEMMKernel",
    "mmqQ8_0WGEMVKernel",
    "mmqQ8_0WFusedKernel",
    "kvAppendQ4_KKernel",
    "kvAppendQ8_KKernel",
    "kvAppendQ2_KKernel",
    "kvAppendIQ4_NLKernel",
    "kvAppendIQ3_XXSKernel",
    "kvAppendMXFP4Kernel",
    "kvAppendIQ2_SKernel",
    "kvAppendTQ2_0Kernel",
    "kvAppendIQ2_XXSKernel",
    "kvAppendIQ2_XSKernel",
    "kvAppendQ3_KKernel",
    "kvAppendIQ4_XSKernel",
    "kvAppendIQ1_SKernel",
    "kvAppendIQ1_MKernel",
    "kvAppendIQ3_SKernel",
    "kvAppendQ8_0Kernel",
    "l2NormHeadsKernel",
    "mropeKernel",
    "mropePosIdsKernel",
    "mropeVisionKernel",
    "layerNormKernel",
    "geluKernel",
    "biasAddKernel",
    "vitAttnHeadKernel",
    "packHeadKernel",
    "unpackHeadKernel",
    "q4gemmM1Kernel",
    "qgemmKernel",
    "rmsNormGateMulKernel",
    "rmsNormKernel",
    "sigmoidGateKernel",
    "sigmoidGateProjKernel",
    "sigmoidKernel",
    "splitQGKernel",
    "swigluKernel",
    "argmaxF32Kernel", // G2 (TODO 1.7): argmax device (warp-shuffle f32→i32)
    "sampleF32GumbelKernel", // 1.15 path-A (lane-a): Gumbel-trick temp>0 + rep_penalty device
    "mergeFeedbackKernel", // RLT: gated recurrent feedback merge
};
var g_funcs: [kernel_names.len]?cudaz.CUfunction = .{null} ** kernel_names.len;

/// Función `kvAppendF16Kernel` (para identificar sus nodos en el grafo).
pub fn kvAppendFunc() ?cudaz.CUfunction {
    for (kernel_names, 0..) |kn, i| {
        if (std.mem.eql(u8, kn, "kvAppendF16Kernel")) return g_funcs[i];
    }
    return null;
}

pub const LayerKernels = struct {
    stream: cudaz.CUstream,
    /// STUDY §5.4: staging device de la cuantización q8_0 de A para MMQ
    /// (aq pad16 | d f32/half según caller | sa). Lazy, crece on-demand.
    mmq_a_buf: cudaz.CUdeviceptr = 0,
    mmq_a_buf_size: usize = 0,
    /// PLAN_MMPROJ Fase B: sections M-RoPE [4]i32 en DEVICE — el kernel
    /// dereferencia el puntero; un array host stack sería illegal address.
    /// Lazy por primer uso; persiste hasta deinit del proceso.
    mrope_sections_buf: cudaz.CUdeviceptr = 0,
    /// G2 (TODO 1.7): salida del argmax device (i32 por fila). Persistente y
    /// PRE-allocado antes de cualquier captura de grafo: un cuMemAlloc dentro
    /// del capture desactiva el grafo en silencio (lección TODO 1.3).
    argmax_out_buf: cudaz.CUdeviceptr = 0,
    /// lane-cuda UC-2.3: ErrorFlag del qgemm piloto — buffer device u32
    /// persistente (mismo patrón capture-safe que argmax_out_buf). El
    /// kernel lo recibe como 8º parámetro (ABI nueva, único launcher).
    ef_buf: cudaz.CUdeviceptr = 0,
    /// Filas para las que está dimensionado `argmax_out_buf`.
    argmax_out_rows: usize = 0,
    // 1.15 path-A (lane-a): sampler Gumbel — counter philox [4]u32 y ring
    // rep_penalty [64]u32 en device, PRE-alocados (capture-safe, 1.3).
    philox_counter_buf: cudaz.CUdeviceptr = 0,
    penalty_ring_buf: cudaz.CUdeviceptr = 0,

    pub fn init(stream: cudaz.CUstream) !LayerKernels {
        _ = try loadModule();
        return .{ .stream = stream };
    }

    pub fn deinit(self: *LayerKernels) void {
        _ = self;
    }

    fn get(self: *LayerKernels, name: [:0]const u8) !cudaz.CUfunction {
        _ = self;
        for (kernel_names, 0..) |kn, i| {
            if (std.mem.eql(u8, kn, name)) {
                if (g_funcs[i]) |f| return f;
                const f = try cudaz.cuModuleGetFunction(try loadModule(), name);
                g_funcs[i] = f;
                return f;
            }
        }
        return cudaz.cuModuleGetFunction(try loadModule(), name) catch return error.KernelNotFound;
    }

    // ── lane-cuda UC-3.2 (piloto): variante TYPE-SAFE de get() ─────────
    // El nombre llega como ENUM comptime (bridge.Bridge sobre kernel_names):
    // un typo (.argmaxF32Kernell) es ERROR DE COMPILACIÓN, no
    // ERROR_NOT_FOUND en runtime (clase del bug ABI dbg). El enum deriva
    // de la MISMA tabla kernel_names — sin duplicación. Cache por índice
    // comptime (mismo slot que get()). Pilotos: argmax + qgemm; migración
    // progresiva del resto de launchers sin big-bang.
    const LKBridge = bridge.Bridge(&kernel_names);

    fn getT(comptime f: LKBridge.Fn) !cudaz.CUfunction {
        comptime std.debug.assert(LKBridge.count == kernel_names.len);
        const idx = comptime @intFromEnum(f);
        if (g_funcs[idx]) |h| return h;
        const name = comptime @tagName(f);
        const h = try cudaz.cuModuleGetFunction(try loadModule(), name);
        g_funcs[idx] = h;
        return h;
    }

    fn n_c(v: usize) c_int {
        return @intCast(v);
    }

    fn n_u(v: usize) c_uint {
        return @intCast(v);
    }

    pub fn rmsNorm(self: *LayerKernels, x: usize, gamma: usize, out: usize, rows: usize, n: usize, eps: f32) !void {
        var xv = x;
        var gv = gamma;
        var ov = out;
        var n1: c_int = n_c(n);
        var eps_c: f32 = eps;
        const func = try self.get("rmsNormKernel");
        var kp = [_]?*anyopaque{ &xv, &gv, &ov, &n1, &eps_c };
        try cudaz.cuLaunchKernel(func, @intCast(rows), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn add(self: *LayerKernels, a: usize, b: usize, out: usize, n: usize) !void {
        var av = a;
        var bv = b;
        var ov = out;
        var n1: c_int = n_c(n);
        const func = try self.get("addKernel");
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// Mul element-wise: out[i] = a[i] · b[i] — wrapper del mulKernel del .cu
    /// (8.3 LFM2.5: gating GDN del ShortConv). Wrapper rescatado para
    /// destrabar el build compartido (kernel .cu ya presente).
    pub fn mul(self: *LayerKernels, a: usize, b: usize, out: usize, n: usize) !void {
        var av = a;
        var bv = b;
        var ov = out;
        var n1: c_int = n_c(n);
        const func = try self.get("mulKernel");
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn addInplace(self: *LayerKernels, a: usize, b: usize, n: usize) !void {
        var av = a;
        var bv = b;
        var n1: c_int = n_c(n);
        const func = try self.get("addInplaceKernel");
        var kp = [_]?*anyopaque{ &av, &bv, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// RLT: Gated recurrent feedback merge.
    /// u_t = e_t + α * σ(W_g [e_t; RMSNorm(s_{t-1})]) ⊙ W_s RMSNorm(s_{t-1})
    /// Single-row kernel: grid=(1,), block=(min(d,256),), smem=3*d*sizeof(f32).
    pub fn mergeFeedback(
        self: *LayerKernels,
        encoder_rep: usize, // [d] e_t
        prev_state: usize, // [d] s_{t-1}
        w_gate: usize, // [d, 2*d] gate projection
        w_state: usize, // [d, d] state projection
        out: usize, // [d] output
        d: usize,
        alpha: f32,
    ) !void {
        var er = encoder_rep;
        var ps = prev_state;
        var wg = w_gate;
        var ws = w_state;
        var ov = out;
        var d_c: c_int = n_c(d);
        var alpha_c: f32 = alpha;
        const func = try self.get("mergeFeedbackKernel");
        const smem_bytes: c_uint = @intCast(3 * d * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &er, &ps, &wg, &ws, &ov, &d_c, &alpha_c };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, @intCast(@min(d, 256)), 1, 1, smem_bytes, self.stream, @ptrCast(&kp), null);
    }

    pub fn swiglu(self: *LayerKernels, gate: usize, up: usize, n: usize) !void {
        var gv = gate;
        var uv = up;
        var n1: c_int = n_c(n);
        const func = try self.get("swigluKernel");
        var kp = [_]?*anyopaque{ &gv, &uv, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn sigmoidGate(self: *LayerKernels, beta: usize, gate: usize, dt_bias: usize, ssm_a: usize, n: usize, dt_rank: usize) !void {
        var bv = beta;
        var gv = gate;
        var dv = dt_bias;
        var sv = ssm_a;
        var n1: c_int = n_c(n);
        var dt1: c_int = n_c(dt_rank);
        const func = try self.get("sigmoidGateKernel");
        var kp = [_]?*anyopaque{ &bv, &gv, &dv, &sv, &n1, &dt1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn sigmoidGateProj(self: *LayerKernels, x: usize, w_beta: usize, w_alpha: usize, dt_bias: usize, ssm_a: usize, beta: usize, gate: usize, N: usize, K: usize, dt_rank: usize) !void {
        var xv = x;
        var wbv = w_beta;
        var wav = w_alpha;
        var dv = dt_bias;
        var sv = ssm_a;
        var bv = beta;
        var gv = gate;
        var N1: c_int = n_c(N);
        var K1: c_int = n_c(K);
        var dt1: c_int = n_c(dt_rank);
        const func = try self.get("sigmoidGateProjKernel");
        var kp = [_]?*anyopaque{ &xv, &wbv, &wav, &dv, &sv, &bv, &gv, &N1, &K1, &dt1 };
        try cudaz.cuLaunchKernel(func, @intCast(dt_rank), @intCast(N), 1, 256, 1, 1, n_u(K * @sizeOf(f32)), self.stream, @ptrCast(&kp), null);
    }

    pub fn l2NormHeads(self: *LayerKernels, conv_out: usize, N: usize, qkv_dim: usize, key_dim: usize, n_k_heads: usize, head_v_dim: usize, eps: f32) !void {
        var cov = conv_out;
        var N1: c_int = n_c(N);
        var qv1: c_int = n_c(qkv_dim);
        var kd1: c_int = n_c(key_dim);
        var nk1: c_int = n_c(n_k_heads);
        var hv1: c_int = n_c(head_v_dim);
        var eps_c: f32 = eps;
        const func = try self.get("l2NormHeadsKernel");
        var kp = [_]?*anyopaque{ &cov, &N1, &qv1, &kd1, &nk1, &hv1, &eps_c };
        try cudaz.cuLaunchKernel(func, @intCast(n_k_heads), @intCast(N), 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn conv1dSilu(self: *LayerKernels, conv_state: usize, qkv: usize, conv_w: usize, conv_out: usize, state_out: usize, N: usize, qkv_dim: usize, d_conv: usize) !void {
        var csv = conv_state;
        var qv = qkv;
        var cwv = conv_w;
        var cov = conv_out;
        var sov = state_out;
        var N1: c_int = n_c(N);
        var qv1: c_int = n_c(qkv_dim);
        var dc1: c_int = n_c(d_conv);
        const func = try self.get("conv1dSiluKernel");
        var kp = [_]?*anyopaque{ &csv, &qv, &cwv, &cov, &sov, &N1, &qv1, &dc1 };
        const rows = if (N > d_conv - 1) N else d_conv - 1;
        try cudaz.cuLaunchKernel(func, n_u((rows * qkv_dim + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// 8.3 (LFM2): conv1d causal LINEAL — sin silu (lfm2.cpp usa
    /// ggml_ssm_conv lineal). Espejo de conv1dSilu con el kernel lineal.
    /// 8.3 (LFM2 GDN): gating per-token sobre in_proj [T, 3*D] interleaved.
    /// mode 0: out = b·x (chunk 0 · chunk 2); mode 1: out = c·conv_out
    /// (chunk 1 del in_proj · buffer conv_out [T, D]).
    pub fn gdnGate(self: *LayerKernels, src: usize, src2: usize, out: usize, T: usize, D: usize, mode: u32) !void {
        var sv = src;
        var s2v = src2;
        var ov = out;
        var T1: c_int = n_c(T);
        var D1: c_int = n_c(D);
        var m1: c_int = @intCast(mode);
        const func = try self.get("gdnGateKernel");
        var kp = [_]?*anyopaque{ &sv, &s2v, &ov, &T1, &D1, &m1 };
        try cudaz.cuLaunchKernel(func, n_u((T * D + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn conv1dLinear(self: *LayerKernels, conv_state: usize, input: usize, conv_w: usize, conv_out: usize, state_out: usize, N: usize, in_dim: usize, d_conv: usize) !void {
        var csv = conv_state;
        var qv = input;
        var cwv = conv_w;
        var cov = conv_out;
        var sov = state_out;
        var N1: c_int = n_c(N);
        var qv1: c_int = n_c(in_dim);
        var dc1: c_int = n_c(d_conv);
        const func = try self.get("conv1dLinearKernel");
        var kp = [_]?*anyopaque{ &csv, &qv, &cwv, &cov, &sov, &N1, &qv1, &dc1 };
        const rows = if (N > d_conv - 1) N else d_conv - 1;
        try cudaz.cuLaunchKernel(func, n_u((rows * in_dim + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn rmsNormGateMul(self: *LayerKernels, attn_out: usize, z: usize, ssm_norm: usize, N: usize, d_inner: usize, n_v_heads: usize, head_v_dim: usize, eps: f32) !void {
        var aov = attn_out;
        var zv = z;
        var snv = ssm_norm;
        var N1: c_int = n_c(N);
        var di1: c_int = n_c(d_inner);
        var nv1: c_int = n_c(n_v_heads);
        var hv1: c_int = n_c(head_v_dim);
        var eps_c: f32 = eps;
        const func = try self.get("rmsNormGateMulKernel");
        var kp = [_]?*anyopaque{ &aov, &zv, &snv, &N1, &di1, &nv1, &hv1, &eps_c };
        try cudaz.cuLaunchKernel(func, @intCast(n_v_heads), @intCast(N), 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// STUDY §5.5: wrapper byte-idéntico de conv1dSilu + l2NormHeads (2 lanzadas).
    pub fn conv1dSiluL2(self: *LayerKernels, conv_state: usize, qkv: usize, conv_w: usize, conv_out: usize, state_out: usize, N: usize, qkv_dim: usize, d_conv: usize, key_dim: usize, n_k_heads: usize, n_v_heads: usize, head_v_dim: usize, eps: f32) !void {
        _ = n_v_heads;
        try self.conv1dSilu(conv_state, qkv, conv_w, conv_out, state_out, N, qkv_dim, d_conv);
        try self.l2NormHeads(conv_out, N, qkv_dim, key_dim, n_k_heads, head_v_dim, eps);
    }

    pub fn deltaNet(self: *LayerKernels, conv_out: usize, gate: usize, beta: usize, attn_out: usize, state: usize, N: usize, qkv_dim: usize, key_dim: usize, n_k_heads: usize, n_v_heads: usize, head_v_dim: usize, dt_rank: usize, eps: f32) !void {
        var cov = conv_out;
        var gv = gate;
        var bv = beta;
        var aov = attn_out;
        var sv = state;
        var N1: c_int = n_c(N);
        var qv1: c_int = n_c(qkv_dim);
        var kd1: c_int = n_c(key_dim);
        var nk1: c_int = n_c(n_k_heads);
        var nv1: c_int = n_c(n_v_heads);
        var hv1: c_int = n_c(head_v_dim);
        var dt1: c_int = n_c(dt_rank);
        var eps_c: f32 = eps;
        const func = try self.get("deltaNetKernel");
        var kp = [_]?*anyopaque{ &cov, &gv, &bv, &aov, &sv, &N1, &qv1, &kd1, &nk1, &nv1, &hv1, &dt1, &eps_c };
        try cudaz.cuLaunchKernel(func, @intCast(n_v_heads), @intCast(N), 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// ΔNet warp-shuffle (STUDY §5.6): 4 columnas/bloque × 32 lanes, árboles
    /// __shfl_down sin __syncthreads, decay fusionado en la actualización.
    /// Requisito: head_v_dim % 32 == 0 (128 en Qwen3.5 ✓). Env `DNWARP=1`
    /// fuerza el kernel clásico (A/B + paridad de referencia).
    pub fn deltaNetWarp(self: *LayerKernels, conv_out: usize, gate: usize, beta: usize, attn_out: usize, state: usize, N: usize, qkv_dim: usize, key_dim: usize, n_k_heads: usize, n_v_heads: usize, head_v_dim: usize, dt_rank: usize, eps: f32) !void {
        var cov = conv_out;
        var gv = gate;
        var bv = beta;
        var aov = attn_out;
        var sv = state;
        var N1: c_int = n_c(N);
        var qv1: c_int = n_c(qkv_dim);
        var kd1: c_int = n_c(key_dim);
        var nk1: c_int = n_c(n_k_heads);
        var nv1: c_int = n_c(n_v_heads);
        var hv1: c_int = n_c(head_v_dim);
        var dt1: c_int = n_c(dt_rank);
        var eps_c: f32 = eps;
        const func = try self.get("deltaNetWarpKernel");
        var kp = [_]?*anyopaque{ &cov, &gv, &bv, &aov, &sv, &N1, &qv1, &kd1, &nk1, &nv1, &hv1, &dt1, &eps_c };
        // Grid: (heads, N, dim/4); block (32, 4, 1).
        try cudaz.cuLaunchKernel(func, @intCast(n_v_heads), @intCast(N), @intCast(head_v_dim / 4), 32, 4, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// STUDY §5.1 FUSED: l2(K/Q) + ΔNet con S-state en REGISTROS (64 KB cero
    /// DRAM en la recurrencia — port unsloth §3.1) + rmsNorm·silu(z) — 3
    /// lanzamientos → 1 por (head, token). Semántica bit-idéntica a la cadena
    /// separada (l2NormHeads + deltaNetWarp + rmsNormGateMul). Requisito:
    /// head_v_dim % 32 == 0. Grid: (n_v_heads, N, 1); block (32, dim/… NO:
    /// block (32, 32, 1) = 1024 threads ⇒ 32 warps = 32 columnas… para
    /// S_v=128 se necesitan 128 columnas ⇒ block (32, 128, 1) excede 1024.
    /// AJUSTE FINAL: block (32, 4, 1) y grid.z = dim/4 — CADA bloque cubre 4
    /// columnas del estado; las etapas A (l2) y C (rmsNorm) necesitan la
    /// head COMPLETA ⇒ sólo el bloque grid.z==0 las ejecuta (guard), con la
    /// desventaja de que las etapas B de otros bloques escriben attn_out
    /// parcialmente. SOLUCIÓN: l2/rms en bloque z==0 tras barrier global —
    /// los otros bloques sólo hacen la recurrencia (etapa B) y el bloque
    /// z==0 hace A(su slice)+B(su slice)+C(completo). VER NOTA de paridad:
    /// C necesita el attn_out de TODAS las columnas ⇒ lanza C como kernel
    /// separado (rmsNormGateMul existe) tras el fused. El fused cubre
    /// A+B (l2 slice + ΔNet slice) por bloque — cada bloque hace SU slice.
    /// STUDY §5.1 FUSED: l2(K/Q) + ΔNet con S-state en REGISTROS (port
    /// unsloth §3.1) + rmsNorm·silu(z) — 3 lanzamientos → 1 por (head, token).
    /// Bit-idéntico a l2NormHeads + deltaNetWarp + rmsNormGateMul (mismo
    /// patrón strided + árbol reds[256] en l2/rms; recurrencia §5.6 exacta).
    /// REQUISITOS: head_v_dim % 32 == 0, dim <= 128, n_v_heads == n_k_heads
    /// (ratio 1 — Qwen3.5-0.8B: 16==16 ✓; con ratio>1 habría carrera
    /// cross-block sobre el l2 compartido de la k-head → caller cae al
    /// camino separado). OPT-IN: DNFUSED=1.
    pub fn deltaNetFused(self: *LayerKernels, conv_out: usize, gate: usize, beta: usize, z: usize, ssm_norm: usize, attn_out: usize, state: usize, N: usize, qkv_dim: usize, key_dim: usize, n_k_heads: usize, n_v_heads: usize, head_v_dim: usize, dt_rank: usize, eps: f32) !void {
        var cov = conv_out;
        var gv = gate;
        var bv = beta;
        var zv = z;
        var snv = ssm_norm;
        var aov = attn_out;
        var sv = state;
        var N1: c_int = n_c(N);
        var qv1: c_int = n_c(qkv_dim);
        var kd1: c_int = n_c(key_dim);
        var nk1: c_int = n_c(n_k_heads);
        var nv1: c_int = n_c(n_v_heads);
        var hv1: c_int = n_c(head_v_dim);
        var dt1: c_int = n_c(dt_rank);
        var eps_c: f32 = eps;
        const func = try self.get("deltaNetFusedKernel");
        var kp = [_]?*anyopaque{ &cov, &gv, &bv, &zv, &snv, &aov, &sv, &N1, &qv1, &kd1, &nk1, &nv1, &hv1, &dt1, &eps_c };
        // Grid: (n_v_heads, N, 1); block (32, 32, 1) = 1024 threads (32 warps,
        // cada warp 4 columnas de las dim=128 — un bloque por head completa).
        try cudaz.cuLaunchKernel(func, @intCast(n_v_heads), @intCast(N), 1, 32, 32, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// STUDY §5.2: chunked batched ΔNet prefill. Procesa K tokens por v-head en
    /// UNA lanzada, manteniendo el estado S_V×S_V en registros a lo largo del
    /// bucle de K tokens. K=64 (default) o K=128. kda=false para Qwen3.5
    /// (decay escalar por cabeza). Requisito: head_v_dim % 32 == 0.
    ///
    /// Layout (el mismo que forwardGPUDecode):
    ///   conv_out: [n, qkv_dim] intercalado q(0)|k(key_dim)|v(2·key_dim)
    ///   gate:     [n, dt_rank]
    ///   beta:     [n, dt_rank]
    ///   state:    [n_v_heads, S_v, S_v] (INOUT)
    ///   attn_out: [n, d_inner]
    pub fn prefillDeltaNetChunk(
        self: *LayerKernels,
        conv_out: usize,
        q_off: c_int,
        k_off: c_int,
        v_off: c_int,
        qkv_stride: c_int,
        gate: usize,
        dt_stride: c_int,
        beta: usize,
        state: usize,
        attn_out: usize,
        d_inner: c_int,
        t_start: c_int,
        scale: f32,
        n_v_heads: c_int,
        n_k_heads: c_int,
        head_v_dim: c_int,
        K: c_int,
        kda: bool,
        n_tokens: c_int, // tokens REALES de este chunk (<= K); evita OOB en chunks parciales
    ) !void {
        const func_name: [:0]const u8 = if (kda)
            if (K == 64) "prefillDeltaNetChunk_kda_K64" else "prefillDeltaNetChunk_kda_K128"
        else if (K == 64) "prefillDeltaNetChunk_nkda_K64" else "prefillDeltaNetChunk_nkda_K128";
        // STUDY §5.2: el prefill chunked vive en su PROPIO cubin (loadPrefillModule),
        // no en el cubin de layer_kernels. Bypasea la caché de funciones.
        const func = try cudaz.cuModuleGetFunction(try loadPrefillModule(), func_name);
        var cov = conv_out;
        var gv = gate;
        var bv = beta;
        var stv = state;
        var aov = attn_out;
        var qo = q_off;
        var ko = k_off;
        var vo = v_off;
        var qvs = qkv_stride;
        var dts = dt_stride;
        var dtsb = dt_stride; // dt_stride_b (beta): mismo valor que gate para Qwen3.5
        var di = d_inner;
        var ts = t_start;
        var sc = scale;
        var nvh = n_v_heads;
        var nkh = n_k_heads;
        var hvd = head_v_dim;
        var nt = n_tokens;
        // 18 args: conv,q_off,k_off,v_off,qkv_stride,gate,dt_stride,beta,dt_stride_b,
        //          state,attn_out,d_inner,t_start,scale,n_v_heads,n_k_heads,head_v_dim,n_tokens
        var kp = [_]?*anyopaque{
            &cov, &qo,  &ko, &vo,   &qvs,
            &gv,  &dts, &bv, &dtsb, &stv,
            &aov, &di,  &ts, &sc,   &nvh,
            &nkh, &hvd, &nt,
        };
        // Grid: (n_v_heads, 1, (S_v+3)/4); block (32, 4, 1).
        try cudaz.cuLaunchKernel(func, @intCast(nvh), 1, @intCast(@divTrunc(hvd + 3, 4)), 32, 4, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// 1.4 §5.2 WY (lane-b): prefill ΔNet por representación WY — 2 kernels
    /// (prefillWYSolve batched + prefillWYState col-parallel). Oráculo:
    /// src/transformer/prefill_wy.zig. Procesa `n_tokens` tokens de UNA vez
    /// (chunks internos de 64); state/attn_out INOUT como en
    /// prefillDeltaNetChunk. Scratch device efímero (8 buffers, se libera
    /// al final — movible a pool si el bench muestra overhead de alloc).
    pub fn prefillWY(
        self: *LayerKernels,
        conv_out: usize,
        q_off: c_int,
        k_off: c_int,
        v_off: c_int,
        qkv_stride: c_int,
        gate: usize,
        dt_stride: c_int,
        beta: usize,
        state: usize,
        attn_out: usize,
        d_inner: c_int,
        scale: f32,
        n_v_heads: c_int,
        n_k_heads: c_int,
        head_v_dim: c_int,
        n_tokens: c_int,
    ) !void {
        if (head_v_dim > 128) return error.DimensionMismatch;
        const CS: usize = 64;
        const S: usize = @intCast(head_v_dim);
        const nvh: usize = @intCast(n_v_heads);
        const n_chunks: usize = (@as(usize, @intCast(n_tokens)) + CS - 1) / CS;

        const mod = try loadPrefillModule();
        const k1 = try cudaz.cuModuleGetFunction(mod, "prefillWYSolve");
        const k2 = try cudaz.cuModuleGetFunction(mod, "prefillWYState");

        // Scratch: g_cs[CS] + attn/kq[CS²] + kg/q_g/avb[CS·S] + k_cd[S·CS] +
        // g_last[1] — todo ×(n_chunks·n_v_heads).
        const per_head2 = n_chunks * nvh;
        const n_g_cs = per_head2 * CS;
        const n_cs2 = per_head2 * CS * CS;
        const n_cs_s = per_head2 * CS * S;
        const n_s_cs = per_head2 * S * CS;
        var d_g_cs = try cudaz.cuMemAlloc(n_g_cs * 4);
        errdefer cudaz.cuMemFree(d_g_cs);
        var d_attn = try cudaz.cuMemAlloc(n_cs2 * 4);
        errdefer cudaz.cuMemFree(d_attn);
        var d_kq = try cudaz.cuMemAlloc(n_cs2 * 4);
        errdefer cudaz.cuMemFree(d_kq);
        var d_kg = try cudaz.cuMemAlloc(n_cs_s * 4);
        errdefer cudaz.cuMemFree(d_kg);
        var d_q_g = try cudaz.cuMemAlloc(n_cs_s * 4);
        errdefer cudaz.cuMemFree(d_q_g);
        var d_k_cd = try cudaz.cuMemAlloc(n_s_cs * 4);
        errdefer cudaz.cuMemFree(d_k_cd);
        var d_avb = try cudaz.cuMemAlloc(n_cs_s * 4);
        errdefer cudaz.cuMemFree(d_avb);
        var d_g_last = try cudaz.cuMemAlloc(per_head2 * 4);
        errdefer cudaz.cuMemFree(d_g_last);
        // 1.4 (lane-f): los frees del scratch DESPUÉS de lanzar K2 deben ser
        // stream-ordenados — cuMemFree inmediato tras el launch libera la
        // VA mientras K1/K2 están en cola (UB CUDA); en E2E la alloc de la
        // capa siguiente robaba la VA y corrompía el scratch pendiente (el
        // test aislado pasaba 4/4 porque su DtoH sí esperaba al stream).
        defer {
            cudaz.cuStreamSynchronize(self.stream) catch {};
            cudaz.cuMemFree(d_g_cs);
            cudaz.cuMemFree(d_attn);
            cudaz.cuMemFree(d_kq);
            cudaz.cuMemFree(d_kg);
            cudaz.cuMemFree(d_q_g);
            cudaz.cuMemFree(d_k_cd);
            cudaz.cuMemFree(d_avb);
            cudaz.cuMemFree(d_g_last);
        }

        var cov = conv_out;
        var gv = gate;
        var bv = beta;
        var stv = state;
        var aov = attn_out;
        var qo = q_off;
        var ko = k_off;
        var vo = v_off;
        var qvs = qkv_stride;
        var dts = dt_stride;
        var dtsb = dt_stride;
        var di = d_inner;
        var sc = scale;
        var nvh1 = n_v_heads;
        var nkh1 = n_k_heads;
        var hvd = head_v_dim;
        var nt = n_tokens;
        var nc: c_int = @intCast(n_chunks);

        var kp1 = [_]?*anyopaque{
            &cov,            &qo,             &ko,              &vo,               &qvs,             &gv,                 &dts,              &bv,
            &dtsb,           &nvh1,           &nkh1,            &hvd,              &nt,              &nc,                 @ptrCast(&d_g_cs), @ptrCast(&d_attn),
            @ptrCast(&d_kq), @ptrCast(&d_kg), @ptrCast(&d_q_g), @ptrCast(&d_k_cd), @ptrCast(&d_avb), @ptrCast(&d_g_last),
        };
        if (debugz.dbg.dump_kv) {
            debugz.dbg.printLevel(.info, "[prefill_wy] K1 ABI: cov={x} qo={d} ko={d} vo={d} qvs={d} gv={x} dts={d} bv={x} dtsb={d} nvh={d} nkh={d} hvd={d} nt={d} nc={d} sc={e} scratch gcs={x} state={x} aov={x}\n", .{ cov, q_off, k_off, v_off, qkv_stride, gv, dt_stride, beta, dt_stride, n_v_heads, n_k_heads, head_v_dim, n_tokens, n_chunks, scale, d_g_cs, state, attn_out });
        }
        try cudaz.cuLaunchKernel(k1, @intCast(n_chunks), @intCast(nvh), 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp1), null);
        if (debugz.dbg.dump_kv) {
            // 1.4 (lane-f): ¿corrompe K1 las ENTRADAS? En E2E el scratch se
            // aloca PEGADO a conv_out (gcs = cov+size exacto) — un write OOB
            // de K1 sería inofensivo en repro (slack de cudaMalloc) pero
            // destructivo en E2E. Compara sumabs de conv pre/post K1.
            try cudaz.cuStreamSynchronize(self.stream);
            const ck = try std.heap.page_allocator.alloc(f32, @as(usize, @intCast(n_tokens)) * @as(usize, @intCast(qkv_stride)));
            defer std.heap.page_allocator.free(ck);
            try cudaz.cuMemcpyDtoH(@intFromPtr(ck.ptr), conv_out, ck.len * 4);
            var s: f64 = 0;
            for (ck) |v| s += @abs(@as(f64, @floatFromInt(@as(i32, @bitCast(v)))));
            debugz.dbg.printLevel(.info, "[prefill_wy] post-K1 conv sumabs={e}\n", .{s});
        }

        var kp2 = [_]?*anyopaque{
            &cov,            &qo,              &ko,               &vo,              &qvs,                &bv, &dtsb, &stv,              &aov,
            &di,             &sc,              &nvh1,             &nkh1,            &hvd,                &nt, &nc,   @ptrCast(&d_attn), @ptrCast(&d_kq),
            @ptrCast(&d_kg), @ptrCast(&d_q_g), @ptrCast(&d_k_cd), @ptrCast(&d_avb), @ptrCast(&d_g_last),
        };
        try cudaz.cuLaunchKernel(k2, @intCast(S), @intCast(nvh), 1, @intCast(S), 1, 1, 0, self.stream, @ptrCast(&kp2), null);
    }

    pub fn copyF32toF16(self: *LayerKernels, src: usize, dst: usize, n: usize) !void {
        var sv = src;
        var dv = dst;
        var n1: c_int = n_c(n);
        const func = try self.get("copyF32toF16Kernel");
        var kp = [_]?*anyopaque{ &sv, &dv, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// C6.1 (5.1, lane-c): GEMV q8_0 del fc dflash con split-K por tiles de
    /// 8192 (smem 32KB) — el M1 genérico (smem k·4B) revienta el límite
    /// sm_86 de 99KB con K=32768 (8 taps × 4096 del 9B).
    pub fn dflashFcGemmM1(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("dflashFcGemmM1Kernel");
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        // 8 warps/bloque → n/8 bloques; 256 thr.
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// C6.1 (5.1, lane-c): gather de taps dflash — concat device-resident de
    /// n_taps punteros (cada → n_embd f32) en out [n_taps·n_embd]. Los
    /// taps_ptrs son un buffer DEVICE de punteros device (los host-ptrs de
    /// launch son illegal-address en el kernel).
    pub fn dflashTapsGather(self: *LayerKernels, taps_ptrs_dev: usize, out: usize, n_embd: usize, n_taps: usize) !void {
        var tp = taps_ptrs_dev;
        var ov = out;
        var ne: c_int = n_c(n_embd);
        var nt: c_int = n_c(n_taps);
        const func = try self.get("dflashTapsGatherKernel");
        var kp = [_]?*anyopaque{ &tp, &ov, &ne, &nt };
        try cudaz.cuLaunchKernel(func, n_u((n_embd + 255) / 256), n_u(n_taps), 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn copyF16toF32(self: *LayerKernels, src: usize, dst: usize, n: usize) !void {
        var sv = src;
        var dv = dst;
        var n1: c_int = n_c(n);
        const func = try self.get("copyF16toF32Kernel");
        var kp = [_]?*anyopaque{ &sv, &dv, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn splitQG(self: *LayerKernels, qg: usize, q: usize, g: usize, N: usize, n_head: usize, head_dim: usize) !void {
        var qgv = qg;
        var qv = q;
        var gv = g;
        var N1: c_int = n_c(N);
        var nh1: c_int = n_c(n_head);
        var hd1: c_int = n_c(head_dim);
        const func = try self.get("splitQGKernel");
        var kp = [_]?*anyopaque{ &qgv, &qv, &gv, &N1, &nh1, &hd1 };
        try cudaz.cuLaunchKernel(func, n_u((N * n_head * head_dim + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// G2 (TODO 1.7): argmax por fila en device — port de `argmax.cu:8-40`
    /// del fork unsloth. `x` [rows, ncols] f32 contiguo (los logits), `dst`
    /// [rows] i32 en device. Un bloque por fila; dentro, reducción
    /// warp-shuffle + shared-memory entre warps.
    ///
    /// Criterio de empate: PRIMERA aparición del máximo (usa `>` estricto),
    /// igual que `Sampler.greedyArgmax` en pipeline.zig — así el token
    /// greedy es idéntico al camino host.
    pub fn argmaxF32(self: *LayerKernels, x: usize, dst: usize, rows: usize, ncols: usize) !void {
        var xv = x;
        var dv = dst;
        var c1: c_int = n_c(ncols);
        const func = try getT(.argmaxF32Kernel); // UC-3.2: typo = compile error
        var kp = [_]?*anyopaque{ &xv, &dv, &c1 };
        // block = min(1024, round_up(ncols, 32)) — ncols (vocab) suele ser
        // ~128K, así que cada hilo barre con stride blockDim.x.
        const want = n_u(((ncols + 31) / 32) * 32);
        const threads: c_uint = if (want > 1024) 1024 else want;
        try cudaz.cuLaunchKernel(func, n_u(rows), 1, 1, threads, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// G2: buffer device persistente para la salida del argmax ([rows] i32).
    /// Debe llamarse ANTES de `beginCapture` (el alloc dentro del capture
    /// desactiva el grafo en silencio — lección TODO 1.3). Crece si hace
    /// falta; nunca se libera hasta el fin del proceso.
    pub fn argmaxOut(self: *LayerKernels, rows: usize) !cudaz.CUdeviceptr {
        if (self.argmax_out_buf != 0 and self.argmax_out_rows >= rows) return self.argmax_out_buf;
        const need = rows * @sizeOf(i32);
        if (self.argmax_out_buf != 0) cudaz.cuMemFree(self.argmax_out_buf);
        self.argmax_out_buf = try cudaz.cuMemAlloc(need);
        self.argmax_out_rows = rows;
        return self.argmax_out_buf;
    }

    // ── 1.15 path-A (lane-a): sampler GPU Gumbel ───────────────────────────
    // Estado device PRE-alocado (capture-safe, lección 1.3): counter philox
    // [4]u32 (campos arriba, junto a argmax_out) y ring de rep_penalty
    // [64]u32. El counter se INICIALIZA con la seed UNA vez (host) y el
    // kernel lo incrementa en device (replay-safe: cada replay consume
    // streams nuevos sin intervención host).

    pub fn sampleGumbelInit(self: *LayerKernels, seed: u64) !void {
        if (self.philox_counter_buf == 0) {
            self.philox_counter_buf = try cudaz.cuMemAlloc(16);
            self.penalty_ring_buf = try cudaz.cuMemAlloc(65 * @sizeOf(u32));
        }
        // Seed → counter (splitmix64 para dispersar bits de seed).
        var z = seed +% 0x9E3779B97F4A7C15;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        z = z ^ (z >> 31);
        const ctr = [_]u32{
            @as(u32, @truncate(z)),
            @as(u32, @truncate(z >> 32)),
            @as(u32, @truncate(z >> 32)) ^ 0x1234567,
            @as(u32, @truncate(z)) ^ 0x89ABCDEF,
        };
        try cudaz.cuMemcpyHtoD(self.philox_counter_buf, @intFromPtr(&ctr), 16);
        const ring0 = [_]u32{0} ** 65;
        try cudaz.cuMemcpyHtoD(self.penalty_ring_buf, @intFromPtr(&ring0), 65 * @sizeOf(u32));
    }

    /// Ring de rep_penalty en device: [0]=n (tamaño), [1..n]=tokens. El
    /// host refresca ANTES de cada replay — el graph captura el PUNTERO,
    /// el contenido cambia por paso (kernel-param congelado no sirve).
    pub fn sampleGumbelSetRing(self: *LayerKernels, tokens: []const u32) !void {
        if (self.penalty_ring_buf == 0) return error.NotInitialized;
        var buf: [65]u32 = @splat(0); // [0]=n + 64 tokens
        const n = @min(tokens.len, 64);
        buf[0] = @intCast(n);
        @memcpy(buf[1 .. 1 + n], tokens[0..n]);
        try cudaz.cuMemcpyHtoD(self.penalty_ring_buf, @intFromPtr(&buf), 65 * @sizeOf(u32));
    }

    /// Muestreo Gumbel: dst[row] = argmax(logits/temp + gumbel) con
    /// rep_penalty scatter del ring (ring[0]=n en device — replay-safe).
    /// `philox_counter`/`ring` de sampleGumbelInit (pre-alocados). Grid
    /// (rows), block como argmax.
    pub fn sampleF32Gumbel(self: *LayerKernels, x: usize, dst: usize, temp: f32, penalty: f32, rows: usize, ncols: usize) !void {
        var xv = x;
        var dv = dst;
        var cv = self.philox_counter_buf;
        var rv = self.penalty_ring_buf;
        var t1 = temp;
        var p1 = penalty;
        var c1: c_int = n_c(ncols);
        const func = try self.get("sampleF32GumbelKernel");
        var kp = [_]?*anyopaque{ &xv, &dv, &cv, &rv, &t1, &p1, &c1 };
        const want = n_u(((ncols + 31) / 32) * 32);
        const threads: c_uint = if (want > 1024) 1024 else want;
        try cudaz.cuLaunchKernel(func, n_u(rows), 1, 1, threads, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn mrope(self: *LayerKernels, data: usize, start_pos: usize, rows: usize, N: usize, head_dim: usize, n_rot: usize, base: f32) !void {
        var dv = data;
        var spv = start_pos;
        var r1: c_int = n_c(rows);
        var N1: c_int = n_c(N);
        var hd1: c_int = n_c(head_dim);
        var nr1: c_int = n_c(n_rot);
        var bc: f32 = base;
        const func = try self.get("mropeKernel");
        var kp = [_]?*anyopaque{ &dv, &spv, &r1, &N1, &hd1, &nr1, &bc };
        try cudaz.cuLaunchKernel(func, n_u((rows + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// MRoPE con pos-ids per-token (PLAN_MMPROJ Fase B): `pos_ids` es un
    /// puntero DEVICE a [N][4] i32 row-major; `sections` [11,11,10,0] style
    /// (subidas a device — el kernel DEREFERENCIA ambos punteros).
    /// Bit-fiel a rope.zig::applyRoPEMultiSectionPosIds (ver kernel .cu).
    pub fn mropePosIds(self: *LayerKernels, data: usize, pos_ids: usize, rows: usize, N: usize, head_dim: usize, n_rot: usize, sections: [4]usize, base: f32) !void {
        var dv = data;
        var pv = pos_ids;
        var r1: c_int = n_c(rows);
        var N1: c_int = n_c(N);
        var hd1: c_int = n_c(head_dim);
        var nr1: c_int = n_c(n_rot);
        // sections → device (lazy; el array local viviría en stack host)
        if (self.mrope_sections_buf == 0) {
            self.mrope_sections_buf = try cudaz.cuMemAlloc(4 * @sizeOf(i32));
        }
        const sec: [4]i32 = .{
            @intCast(sections[0]),
            @intCast(sections[1]),
            @intCast(sections[2]),
            @intCast(sections[3]),
        };
        try cudaz.cuMemcpyHtoD(self.mrope_sections_buf, @intFromPtr(&sec), 4 * @sizeOf(i32));
        var sec_dev = self.mrope_sections_buf;
        var bc: f32 = base;
        const func = try self.get("mropePosIdsKernel");
        var kp = [_]?*anyopaque{ &dv, &pv, &r1, &N1, &hd1, &nr1, &sec_dev, &bc };
        try cudaz.cuLaunchKernel(func, n_u((rows + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// M-RoPE VISION interleaved (PLAN_MMPROJ 10.2-bisect, lane-mmproj):
    /// pares adyacentes (2i, 2i+1), freq 10000^(-2·ip/n_pairs),
    /// sector = ip % sect_dims — port bit-fiel de mrope_vision.applyMRopeVision
    /// (el ViT NO usa NEOX half-split). `data`: pack por head [n_head·N, hd].
    pub fn mropeVision(self: *LayerKernels, data: usize, pos_ids: usize, rows: usize, N: usize, head_dim: usize, sections: [4]usize) !void {
        var dv = data;
        var pv = pos_ids;
        var r1: c_int = n_c(rows);
        var N1: c_int = n_c(N);
        var hd1: c_int = n_c(head_dim);
        if (self.mrope_sections_buf == 0) {
            self.mrope_sections_buf = try cudaz.cuMemAlloc(4 * @sizeOf(i32));
        }
        const sec: [4]i32 = .{
            @intCast(sections[0]),
            @intCast(sections[1]),
            @intCast(sections[2]),
            @intCast(sections[3]),
        };
        try cudaz.cuMemcpyHtoD(self.mrope_sections_buf, @intFromPtr(&sec), 4 * @sizeOf(i32));
        var sec_dev = self.mrope_sections_buf;
        const func = try self.get("mropeVisionKernel");
        var kp = [_]?*anyopaque{ &dv, &pv, &r1, &N1, &hd1, &sec_dev };
        try cudaz.cuLaunchKernel(func, n_u((rows + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    // ─── ViT device kernels (PLAN_MMPROJ 10.2, lane-mmproj) ────────────────

    /// LayerNorm device (ViT): rows = n_pos, n = n_embd. gamma/beta device.
    /// Port GPU de clip_block.layerNormSlice.
    pub fn layerNormDev(self: *LayerKernels, x: usize, gamma: usize, beta: usize, out: usize, rows: usize, n: usize, eps: f32) !void {
        var xv = x;
        var gv = gamma;
        var bv = beta;
        var ov = out;
        var n1: c_int = n_c(n);
        var ec: f32 = eps;
        const func = try self.get("layerNormKernel");
        var kp = [_]?*anyopaque{ &xv, &gv, &bv, &ov, &n1, &ec };
        try cudaz.cuLaunchKernel(func, n_u(rows), 1, 1, n_u(@min(n, 256)), 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// GELU tanh in-place device (ViT FFN): x[i] = gelu(x[i]).
    /// Kernel firma (x, out, n) — in-place con out == x.
    pub fn geluDev(self: *LayerKernels, x: usize, n: usize) !void {
        var xv = x;
        var ov = x; // in-place
        var n1: c_int = n_c(n);
        const func = try self.get("geluKernel");
        var kp = [_]?*anyopaque{ &xv, &ov, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// Bias add in-place: x[i] += b[i % b_len] (device).
    pub fn biasAddDev(self: *LayerKernels, x: usize, b: usize, n: usize, b_len: usize) !void {
        var xv = x;
        var bv = b;
        var n1: c_int = n_c(n);
        var bl1: c_int = n_c(b_len);
        const func = try self.get("biasAddKernel");
        var kp = [_]?*anyopaque{ &xv, &bv, &n1, &bl1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// Pack por head (10.2): interleaved [n_pos, n_head, hd] → head-major
    /// [n_head][n_pos·hd]. src apunta al segmento (q/k/v) del qkv.
    pub fn packHead(self: *LayerKernels, src: usize, dst: usize, n_pos: usize, n_head: usize, hd: usize, src_row_stride: usize) !void {
        var sv = src;
        var dv = dst;
        var np1: c_int = n_c(n_pos);
        var nh1: c_int = n_c(n_head);
        var hd1: c_int = n_c(hd);
        var rs1: c_int = n_c(src_row_stride);
        const func = try self.get("packHeadKernel");
        var kp = [_]?*anyopaque{ &sv, &dv, &np1, &nh1, &hd1, &rs1 };
        const total = n_head * n_pos * hd;
        try cudaz.cuLaunchKernel(func, n_u((total + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// Unpack por head (inverso del pack): head-major → interleaved.
    pub fn unpackHead(self: *LayerKernels, src: usize, dst: usize, n_pos: usize, n_head: usize, hd: usize) !void {
        var sv = src;
        var dv = dst;
        var np1: c_int = n_c(n_pos);
        var nh1: c_int = n_c(n_head);
        var hd1: c_int = n_c(hd);
        const func = try self.get("unpackHeadKernel");
        var kp = [_]?*anyopaque{ &sv, &dv, &np1, &nh1, &hd1 };
        const total = n_head * n_pos * hd;
        try cudaz.cuLaunchKernel(func, n_u((total + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// Atención ViT por head device (bidireccional). q/k/v/o [n_pos, hd]
    /// contiguos del head. Un block por query; blockDim=128 (hd ≤ 128).
    pub fn vitAttnHead(self: *LayerKernels, q: usize, k: usize, v: usize, o: usize, n_pos: usize, hd: usize) !void {
        var qv = q;
        var kv = k;
        var vv = v;
        var ov = o;
        var np1: c_int = n_c(n_pos);
        var hd1: c_int = n_c(hd);
        var kq_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const func = try self.get("vitAttnHeadKernel");
        var kp = [_]?*anyopaque{ &qv, &kv, &vv, &ov, &np1, &hd1, &kq_scale };
        try cudaz.cuLaunchKernel(func, n_u(n_pos), 1, 1, 128, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn kvAppendF16(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        const func = try self.get("kvAppendF16Kernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1 };
        try cudaz.cuLaunchKernel(func, n_u((n * kv_dim + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// KV-append q8_0: cuantiza K/V float→q8_0 canónico (34 B por grupo de 32,
    /// escala f16 embebida) al pool paginado. Un solo CUDA-block recorre en
    /// serie bloques físicos → K/V → grupos (los grupos de 34B comparten
    /// sectores de 32B entre regiones; escribir desde varios blocks pierde
    /// actualizaciones — ver comentario del kernel). Sin requisito de
    /// alineación de kv_dim.
    pub fn kvAppendQ8_0(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        // Cada grupo de 32 elems debe caer íntegro en un token para poder
        // cuantizarlo desde el chunk actual sin leer tokens previos.
        std.debug.assert(kv_dim % 32 == 0);
        if (debugz.dbg.at(.detail)) {
            debugz.dbg.print("[kv-append-q8_0] k={x} v={x} cache={x} bt={x} sp={x} n={d} kv_dim={d} n_kv_head={d} head_dim={d} block_size={d}\n", .{ k, v, cache, bt, start_pos, n, kv_dim, n_kv_head, head_dim, block_size });
        }
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendQ8_0Kernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        // Breadcrumb DUMP_KVQUANT: sincroniza y reporta fallos asíncronos del
        // kernel aquí en vez de dejar que envenenen el contexto y exploten en
        // un sync lejano (ERROR_LAUNCH_FAILED 700 engañoso).
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-q8_0] FALLO asíncrono tras launch: {s} (sp={d} n={d} cache={x} bt={x})\n", .{ @errorName(e), start_pos, n, cache, bt });
                return e;
            };
        }
    }

    /// KV-append q4_K: cuantiza K/V float→q4_K canónico (144 B por SB de 256,
    /// d/dmin f16 + escalas 6-bit empaquetadas + nibbles). Requiere
    /// kv_dim % 256 == 0 (SB íntegro en un token). Mismo patrón serializado.
    pub fn kvAppendQ4_K(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 256 == 0);
        if (debugz.dbg.at(.detail)) {
            debugz.dbg.print("[kv-append-q4_k] k={x} v={x} cache={x} bt={x} sp={x} n={d} kv_dim={d}\n", .{ k, v, cache, bt, start_pos, n, kv_dim });
        }
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendQ4_KKernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-q4_k] FALLO asíncrono tras launch: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// Garantiza mmq_a_buf ≥ `bytes`, creciendo si el cacheado no llega.
    /// (Lección 1.10: el buffer se comparte entre LMSPLIT M=1 y WMMA M≥128
    /// — un tamaño congelado por el primer caller producía OOB write device.)
    fn ensureMmqABuf(self: *LayerKernels, bytes: usize) !void {
        if (self.mmq_a_buf != 0 and self.mmq_a_buf_size >= bytes) return;
        if (self.mmq_a_buf != 0) cudaz.cuMemFree(self.mmq_a_buf);
        self.mmq_a_buf = try cudaz.cuMemAlloc(bytes);
        self.mmq_a_buf_size = bytes;
    }

    /// B3 MMQ paso 1: cuantiza A[M,K] f32 a q8_0 por bloques de 32
    /// (a_i8 + escala f16 + Σq i32 para la corrección unsigned-nibble).
    pub fn mmqQuantizeA(self: *LayerKernels, a: usize, aq: usize, ad: usize, asa: usize, m: usize, k: usize) !void {
        var av = a;
        var qv = aq;
        var dv = ad;
        var sv = asa;
        var m1: c_int = n_c(m);
        var k1: c_int = n_c(k);
        const func = try self.get("mmqQuantizeAQ8Kernel");
        var kp = [_]?*anyopaque{ &av, &qv, &dv, &sv, &m1, &k1 };
        try cudaz.cuLaunchKernel(func, n_u(m * (k / 32)), 1, 1, 32, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[mmq-quantA] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// B3 MMQ paso 2: C[M,N] = A_q8 · W_q4_0 vía dp4a. M ≤ 4.
    pub fn mmqQ4_0GEMV(self: *LayerKernels, aq: usize, ad: usize, asa: usize, w: usize, c: usize, m: usize, k: usize, nn: usize) !void {
        std.debug.assert(m <= 4);
        std.debug.assert(k % 32 == 0);
        var av = aq;
        var dv = ad;
        var sv = asa;
        var wv = w;
        var cv = c;
        var m1: c_int = n_c(m);
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(nn);
        const kb_total = k / 32;
        // smem: aq [M*K] pad16 | d f32 [M*KB] | sa f32 [M*KB]
        const aq_bytes = (m * k + 15) & ~@as(usize, 15);
        const smem: c_uint = @intCast(aq_bytes + m * kb_total * 4 * 2);
        const func = try self.get("mmqQ4_0GEMVKernel");
        var kp = [_]?*anyopaque{ &av, &dv, &sv, &wv, &cv, &m1, &k1, &n1 };
        // C pre-cero (los slices acumulan con atomics)
        try cudaz.cuMemsetD8(cv, 0, m * nn * @sizeOf(f32));
        try cudaz.cuLaunchKernel(func, n_u((nn + 7) / 8), 1, 4, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[mmq-gemv] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// MMQ Q4_0 GEMM — tensor-core WMMA INT8 for prefill M>=128.
    /// Stage 1: quantize A to q8_0 via mmqQuantizeAQ8Kernel.
    /// Stage 2: dequant q4_0 B to int8 in shared mem + WMMA INT8 (2 mma_sync per tile for BK=32).
    pub fn mmqQ4_0GEMM(self: *LayerKernels, aq: usize, ad: usize, asa: usize, w: usize, c: usize, m: usize, k: usize, nn: usize) !void {
        std.debug.assert(m >= 128);
        std.debug.assert(k % 32 == 0);
        var av = aq;
        var dv = ad;
        var sv = asa;
        var wv = w;
        var cv = c;
        var m1: c_int = n_c(m);
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(nn);
        const func = try self.get("mmqQ4_0GEMMKernel");
        // Shared (kernel layout): s_aq[128*48 i8] | s_ad[128 f32] |
        // s_bwr[128*32 i8] | s_dB[128 f32] | s_cW[8 warps × 256 i32]
        const s_aq: usize = 128 * 48;
        const s_ad: usize = 128 * @sizeOf(f32);
        const s_bwr: usize = 128 * 32;
        const s_dB: usize = 128 * @sizeOf(f32);
        const s_cW: usize = 8 * 256 * @sizeOf(i32);
        const smem: c_uint = @intCast(s_aq + s_ad + s_bwr + s_dB + s_cW);
        // C se escribe exactamente una vez por posición (tiles de warp
        // disjuntos 64×32) — NO requiere memset previo.
        const nwarps: usize = 8;
        try cudaz.cuLaunchKernel(func, n_u((nn + 127) / 128), n_u((m + 127) / 128), 1, nwarps * 32, 1, 1, smem, self.stream, @ptrCast(@constCast(&[_]?*anyopaque{ &av, &dv, &sv, &wv, &cv, &m1, &k1, &n1 })), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[mmq-gemm] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// KV-append IQ1_S: cuantiza K/V float→IQ1_S canónico (50B por SB de
    /// 256: d f16 + qs[32] + qh[16]). Brute-force serial en tid==0.
    /// Requiere kv_dim % 256 == 0. Mismo patrón serializado.
    pub fn kvAppendIQ1_S(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 256 == 0);
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendIQ1_SKernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        // 16 warps ↔ combos (delta,neg); scan gi repartido entre lanes.
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 512, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-iq1_s] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// KV-append IQ1_M: cuantiza K/V float→IQ1_M layout entrelazado KV-path
    /// (56B por SB de 256: sc16 pares solapados con qb[0..8) + nibbles-d en
    /// altos de bytes impares + qh@32; padding [48..56)=0). Búsqueda
    /// cooperativa 512 hilos (candidato/hilo en grupos libres ib≥2; ib<2 con
    /// qb impuesto). Espejo encodeIQ1_M. Requiere kv_dim % 256 == 0.
    pub fn kvAppendIQ1_M(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 256 == 0);
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendIQ1_MKernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        // 512 hilos ↔ candidatos qb×dd de los grupos libres.
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 512, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-iq1_m] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// KV-append IQ3_S: cuantiza K/V float→IQ3_S canónico (110B por SB de
    /// 256: d f16 + qs[64] + qh[8] + signs[32] + scales[4]). Brute-force
    /// serial en tid==0 (espejo encodeIQ3_S). Requiere kv_dim % 256 == 0.
    pub fn kvAppendIQ3_S(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 256 == 0);
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendIQ3_SKernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        // 16 warps ↔ códigos sc; scan gi repartido entre lanes.
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 512, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-iq3_s] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// KV-append IQ2_XXS: cuantiza K/V float→IQ2_XXS canónico (66B por SB
    /// de 256: d f16 + qs[64] con aux0/aux1 por sub-bloque). Serial tid==0
    /// espejo encodeIQ2_XXS. Requiere kv_dim % 256 == 0.
    pub fn kvAppendIQ2_XXS(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 256 == 0);
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendIQ2_XXSKernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-iq2_xxs] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// KV-append IQ2_XS: cuantiza K/V float→IQ2_XS canónico (74B por SB
    /// de 256: d f16 + qs[64] v-u16 + scales[8] nibble). Serial tid==0
    /// espejo encodeIQ2_XS. Requiere kv_dim % 256 == 0.
    pub fn kvAppendIQ2_XS(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 256 == 0);
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendIQ2_XSKernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-iq2_xs] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// KV-append MXFP4: cuantiza K/V float→MXFP4 canónico (17B/bloque32:
    /// escala u8 E8M0 + qs[16] split-16). Gran 32 ⇒ kv_dim % 32 == 0.
    pub fn kvAppendMXFP4(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 32 == 0);
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendMXFP4Kernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-mxfp4] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// KV-append TQ2_0: cuantiza K/V float→TQ2_0 canónico (66B/SB256).
    /// Serial warp-reduce d + serial quanta. Requiere kv_dim % 256 == 0.
    pub fn kvAppendTQ2_0(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 256 == 0);
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendTQ2_0Kernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                return e;
            };
        }
    }

    /// KV-append TQ1_0: cuantiza K/V float→TQ1_0 canónico (54B/SB256,
    /// [qs 32][qs2 16][qh 4][d f16@52]). Serial tid==0 espejo encodeTQ1_0
    /// (pack base-3 con LUT inversa runtime, espejo de TQ1_LUT5/LUT4
    /// comptime). Requiere kv_dim % 256 == 0. (6.3 lane-f)
    pub fn kvAppendTQ1_0(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 256 == 0);
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendTQ1_0Kernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                return e;
            };
        }
    }

    /// KV-append IQ2_S: cuantiza K/V float→IQ2_S canónico (82B por SB de
    /// 256: d f16 + qs[32] + signs[32] libres + qh[8] + scales[8] nibble).
    /// Serial tid==0 espejo encodeIQ2_S. Requiere kv_dim % 256 == 0.
    pub fn kvAppendIQ2_S(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 256 == 0);
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendIQ2_SKernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        // F3-next (lane-f): MULTI-BLOCK — sb = blockIdx*8 + warp. Con
        // gridDim=1 TODO el kernel corría en UN SM (f64 1-SM ≈ 33ms, el
        // 24.9-35.5ms medido). grid = ceil(sb/8) blocks reparte los warps
        // por todos los SMs.
        const sb_per_region = (block_size * kv_dim + 255) / 256;
        const blocks = (sb_per_region + 7) / 8;
        try cudaz.cuLaunchKernel(func, n_u(blocks), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-iq2_s] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// KV-append IQ3_XXS: cuantiza K/V float→IQ3_XXS canónico (98B por SB
    /// de 256: d f16 + qs[64] + ss[32] con sc/idx-signos empaquetados).
    /// Serial tid==0 espejo encodeIQ3_XXS. Requiere kv_dim % 256 == 0.
    pub fn kvAppendIQ3_XXS(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 256 == 0);
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendIQ3_XXSKernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-iq3_xxs] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// KV-append IQ4_NL: cuantiza K/V float→IQ4_NL canónico (18B/bloque32:
    /// [d f16][qs16 split-16], valor=d·kvalues[nibble]). Gran 32 ⇒ solo
    /// exige kv_dim % 32 == 0. Guard continue por grupo fuera-del-chunk.
    pub fn kvAppendIQ4_NL(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 32 == 0);
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendIQ4_NLKernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-iq4_nl] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// KV-append Q2_K: cuantiza K/V float→Q2_K canónico (84B por SB de 256:
    /// scales[16] + qs[64] + d/min f16). Serial tid==0 espejo encodeQ2_K.
    /// Requiere kv_dim % 256 == 0. Guard preservación SB íntegro.
    pub fn kvAppendQ2_K(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 256 == 0);
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendQ2_KKernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-q2_k] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// KV-append Q3_K: cuantiza K/V float→Q3_K canónico (110B por SB de 256:
    /// hmask[32] + qs[64] + scales[12] reordenadas + d f16). Serial tid==0
    /// espejo encodeQ3_K (inversa kmask incluida). Requiere kv_dim % 256 == 0.
    pub fn kvAppendQ3_K(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 256 == 0);
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendQ3_KKernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-q3_k] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// KV-append IQ4_XS: cuantiza K/V float→IQ4_XS canónico (136B por SB de
    /// 256: d f16 + scales_h u16 + scales_l[4] pares nibble + qs[128] LUT).
    /// Requiere kv_dim % 256 == 0. Mismo patrón serializado.
    pub fn kvAppendIQ4_XS(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 256 == 0);
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendIQ4_XSKernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-iq4_xs] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// B6 — GEMV con PESOS q8_0 (lm_head cuantizado on-load): C[M,N] = A_q8·Wᵀ.
    /// M ≤ 16 (decode + drafts). Split-K z=4, atomics, C pre-cero.
    pub fn mmqQ8_0WGEMV(self: *LayerKernels, aq: usize, ad: usize, w: usize, c: usize, m: usize, k: usize, nn: usize) !void {
        std.debug.assert(m <= 8);
        std.debug.assert(k % 32 == 0);
        var av = aq;
        var dv = ad;
        var wv = w;
        var cv = c;
        var m1: c_int = n_c(m);
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(nn);
        const kb_total = k / 32;
        const aq_bytes = (m * k + 15) & ~@as(usize, 15);
        const smem: c_uint = @intCast(aq_bytes + m * kb_total * 4);
        const func = try self.get("mmqQ8_0WGEMVKernel");
        // Breadcrumb DUMP_KVQUANT: traza por-bloque del GEMV (kb/lane objetivo).
        var dbg_on: c_int = @intFromBool(debugz.dbg.dump_kvquant);
        var dbg_kb: c_int = blk: {
            const v = std.c.getenv("MMQ_DBG_KB");
            if (v) |p| {
                if (std.fmt.parseInt(c_int, std.mem.span(p), 10)) |x| break :blk x else |_| {}
            }
            break :blk -1;
        };
        var dbg_lane: c_int = blk: {
            const v = std.c.getenv("MMQ_DBG_LANE");
            if (v) |p| {
                if (std.fmt.parseInt(c_int, std.mem.span(p), 10)) |x| break :blk x else |_| {}
            }
            break :blk 0;
        };
        var kp = [_]?*anyopaque{ &av, &dv, &wv, &cv, &m1, &k1, &n1, &dbg_on, &dbg_kb, &dbg_lane };
        try cudaz.cuMemsetD8(cv, 0, m * nn * @sizeOf(f32));
        try cudaz.cuLaunchKernel(func, n_u((nn + 7) / 8), 1, 4, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[mmq-q80w] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// B3-v3 — GEMV q8_0W FUSIONADO: cuantiza A en-kernel (fase 1 smem, solo
    /// el rango kb de cada split) + GEMV dp4a en UN launch (sin quantizeA ni
    /// roundtrip global aq/ad). C pre-cero + split-K z=4 atomics igual que
    /// mmqQ8_0WGEMV. M ≤ 8.
    pub fn mmqQ8_0WFused(self: *LayerKernels, a: usize, w: usize, c: usize, m: usize, k: usize, nn: usize) !void {
        std.debug.assert(m <= 8);
        std.debug.assert(k % 32 == 0);
        var av = a;
        var wv = w;
        var cv = c;
        var m1: c_int = n_c(m);
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(nn);
        const kb_total = k / 32;
        const aq_bytes = (m * k + 15) & ~@as(usize, 15);
        const smem: c_uint = @intCast(aq_bytes + m * kb_total * 4);
        const func = try self.get("mmqQ8_0WFusedKernel");
        var dbg_on: c_int = @intFromBool(debugz.dbg.dump_kvquant);
        var dbg_kb: c_int = -1;
        var dbg_lane: c_int = 0;
        var kp = [_]?*anyopaque{ &av, &wv, &cv, &m1, &k1, &n1, &dbg_on, &dbg_kb, &dbg_lane };
        try cudaz.cuMemsetD8(cv, 0, m * nn * @sizeOf(f32));
        try cudaz.cuLaunchKernel(func, n_u((nn + 7) / 8), 1, 4, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[mmq-fused] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// KV-append q8_K: cuantiza K/V float→q8_K canónico (292B por SB de 256:
    /// d f32 LE + qs i8×256 + bsums ceros). Requiere kv_dim % 256 == 0.
    pub fn kvAppendQ8_K(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 256 == 0);
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendQ8_KKernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-q8_k] FALLO asíncrono: {s}\n", .{@errorName(e)});
                return e;
            };
        }
    }

    /// KV-append q4_0: cuantiza K/V float→q4_0 canónico (18 B por grupo de 32,
    /// escala f16 embebida + nibbles lo/hi) al pool paginado. Mismo patrón
    /// serializado que q8_0 (contrato single-writer por sector). Requiere
    /// kv_dim % 32 == 0 (grupo íntegro en un token).
    pub fn kvAppendQ4_0(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, start_pos: usize, n: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        std.debug.assert(kv_dim % 32 == 0);
        if (debugz.dbg.at(.detail)) {
            debugz.dbg.print("[kv-append-q4_0] k={x} v={x} cache={x} bt={x} sp={x} n={d} kv_dim={d}\n", .{ k, v, cache, bt, start_pos, n, kv_dim });
        }
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var spv = start_pos;
        var n1: c_int = n_c(n);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        var dbg1: c_int = @intCast(debugz.dbg.dump_kvquant_val);
        const func = try self.get("kvAppendQ4_0Kernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &spv, &n1, &kvd1, &nkh1, &hd1, &bs1, &dbg1 };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(self.stream) catch |e| {
                debugz.dbg.print("[kv-append-q4_0] FALLO asíncrono tras launch: {s} (cache={x} bt={x})\n", .{ @errorName(e), cache, bt });
                return e;
            };
        }
    }

    pub fn gateMul(self: *LayerKernels, attn: usize, g: usize, n: usize) !void {
        var av = attn;
        var gv = g;
        var n1: c_int = n_c(n);
        const func = try self.get("gateKernel");
        var kp = [_]?*anyopaque{ &av, &gv, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn embeddingGather(self: *LayerKernels, emb: usize, token: u32, out: usize, n_embd: usize) !void {
        var ev = emb;
        var ov = out;
        var tok1: c_int = n_c(token);
        var ne1: c_int = n_c(n_embd);
        const func = try self.get("embeddingGatherKernel");
        var kp = [_]?*anyopaque{ &ev, &tok1, &ov, &ne1 };
        try cudaz.cuLaunchKernel(func, n_u((n_embd + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// GEMM Q4_0 M=1: C[1,N] = A[1,K] * B_q4[K,N] (peso en bytes Q4_0 tal cual
    /// el GGUF, sin dequantizar; K múltiplo de 32). Un warp por fila de salida:
    /// 256 hilos = 8 filas por bloque → grid = ceil(N/8).
    pub fn q4gemmM1(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("q4gemmM1Kernel");
        const smem: c_uint = @intCast(k * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// STUDY 1.1: M=1 q5_k GEMV — ssm_out del 0.8B (2048→1024). Warp=fila;
    /// lane cubre 2 elems por grupo (nib bajo/alto del mismo byte de qs —
    /// espejo EXACTO de lane_q5k_val del qgemmKernel case 2, misma aritmética
    /// FMA escalar; el win vs qgemm: smem de A compartido entre las 8 filas
    /// del bloque + shfl único). Requisito: K % 256 == 0 (SB Q5_K).
    pub fn q5gemmM1(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 256 == 0);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("q5gemmM1Kernel");
        const smem: c_uint = @intCast(k * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// STUDY 1.2: M=1 q6_k GEMV — lm_head/FFN-down. Warp=fila; lane fija il
    /// y recorre (ip,j) — 8 elems/lane/SB. Espejo EXACTO de lane_q6k_val
    /// (case 3 del qgemmKernel); win = smem de A compartido. K % 256 == 0.
    pub fn q6gemmM1(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 256 == 0);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("q6gemmM1Kernel");
        const smem: c_uint = @intCast(k * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// U3 (eje §12): M=1 q4_k GEMV — cierra EL hueco del Q4_K_XL (97% del
    /// token era SSM con proyecciones q4_k por el camino GENÉRICO). Warp=
    /// fila; lane cubre los 32 elems del sub-bloque con escalas uniformes
    /// en el warp (unpack ramificado POR SUB-BLOQUE, no por elemento).
    /// Espejo EXACTO del case 4 del qgemmKernel — bit-paridad por
    /// construcción. Requisito: K % 256 == 0 (SB Q4_K de 256 elems).
    pub fn q4kGemmM1(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 256 == 0);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("q4kGemmM1Kernel");
        const smem: c_uint = @intCast(k * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// U3 (eje §12): M=1 q8_0 GEMV — 2º hueco del Q4_K_XL (bancos ssm_*
    /// ΔNet q8_0 con K=2048). Espejo EXACTO del case 5 del qgemmKernel.
    pub fn q8kGemmM1(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 32 == 0);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("q8kGemmM1Kernel");
        const smem: c_uint = @intCast(k * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// P0-6: M=1 q4_1 GEMV dp4a. Layout 20B/SB256 [d f16][mm f16][qs64].
    /// Espejo q4_0 dp4a con término extra +mm*A. K%32==0.
    pub fn q41GemmM1Dp4a(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 32 == 0);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("q41GemmM1Dp4aKernel");
        const kb_total = k / 32;
        const aq_pad = (k + 15) & ~@as(usize, 15);
        const smem: c_uint = @intCast(aq_pad + 2 * kb_total * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// P0-6: M=1 q2_k GEMV dp4a. Layout 84B/SB256 [scales16][qs64][d f16@80][min f16@82].
    /// Espejo q3_k dp4a con scales q2_k. K%256==0.
    pub fn q2kGemmM1Dp4a(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 256 == 0);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("q2kGemmM1Dp4aKernel");
        const kb = k / 32;
        const smem: c_uint = @intCast(((k + 15) & ~@as(usize, 15)) + kb * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// P0-6: M=1 iq3_s GEMV dp4a. Layout 110B/SB256 [d f16][qs64][qh8][signs32][scales4].
    /// K%256==0.
    pub fn iq3sGemmM1Dp4a(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 256 == 0);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("iq3sGemmM1Dp4aKernel");
        const kb = k / 32;
        const smem: c_uint = @intCast(((k + 15) & ~@as(usize, 15)) + kb * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// P0-6: M=1 iq2_s GEMV dp4a. Layout 82B/SB256 [d f16][qs32][signs32][qh8][scales8].
    /// K%256==0.
    pub fn iq2sGemmM1Dp4a(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 256 == 0);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("iq2sGemmM1Dp4aKernel");
        const kb = k / 32;
        const smem: c_uint = @intCast(((k + 15) & ~@as(usize, 15)) + kb * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// P0-6: M=1 iq4_xs GEMV dp4a. Layout 136B/SB256 [d f16][scales_h u16][scales_l][qs128].
    /// K%256==0.
    pub fn iq4xsGemmM1Dp4a(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 256 == 0);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("iq4xsGemmM1Dp4aKernel");
        const kb = k / 32;
        const smem: c_uint = @intCast(((k + 15) & ~@as(usize, 15)) + kb * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// P0-6: M=1 iq4_nl GEMV dp4a. Layout 18B/bloque32 [d f16][qs16 split-16].
    /// K%32==0.
    pub fn iq4nlGemmM1Dp4a(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 32 == 0);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("iq4nlGemmM1Dp4aKernel");
        const kb = k / 32;
        const smem: c_uint = @intCast(((k + 15) & ~@as(usize, 15)) + kb * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// P0-6: M=1 iq3_xxs GEMV dp4a. Layout 98B/SB256 [d f16][qs64][ss32].
    /// K%256==0.
    pub fn iq3xxsGemmM1Dp4a(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 256 == 0);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("iq3xxsGemmM1Dp4aKernel");
        const kb = k / 32;
        const smem: c_uint = @intCast(((k + 15) & ~@as(usize, 15)) + kb * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// P0-6: M=1 iq2_xxs GEMV dp4a. Layout 66B/SB256 [d f16][qs64].
    /// K%256==0.
    pub fn iq2xxsGemmM1Dp4a(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 256 == 0);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("iq2xxsGemmM1Dp4aKernel");
        const kb = k / 32;
        const smem: c_uint = @intCast(((k + 15) & ~@as(usize, 15)) + kb * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// P0-6: M=1 iq2_xs GEMV dp4a. Layout 74B/SB256 [d f16][qs64][scales8].
    /// K%256==0.
    pub fn iq2xsGemmM1Dp4a(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 256 == 0);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("iq2xsGemmM1Dp4aKernel");
        const kb = k / 32;
        const smem: c_uint = @intCast(((k + 15) & ~@as(usize, 15)) + kb * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// a-U3 (eje §12): M=1 q3_k GEMV dp4a — EL cuello del gate 3B
    /// (Llama-3.2-3B-Q3_K_S todo q3_k: case 6 escalar = 712µs/capa,
    /// 16% techo HBM). Formulación vec_dot_q3_K_q8_1 de llama.cpp sobre
    /// nuestro layout 110B/SB256: A→q8_1 en smem (fase 1), lane=(j,chunk)
    /// con 2 dp4a/SB — el término hmask se resta DENTRO del dp4a
    /// (__vsubss4 + ~hmask, bytes −4..3). ~2.5 instr/elem vs ~10.
    /// Requisito: K % 256 == 0 (SB q3_k). Shared: aq pad16 | d8 f32[KB].
    /// Fase 2 (split-K): S warps/fila con SBs interleaved. INTER-block
    /// (S>0, más bloques, re-paga fase 1 ×S) fue NEGATIVO: 0.74–0.92×
    /// (microbench 2026-09-10) ⇒ uso INTRA-block (S<0: los 8 warps del
    /// bloque cubren 8/S filas en S partes; fase 1 se paga 1×, critical
    /// path por fila ÷S). Con |S|>1 el buffer c se pre-inicializa con
    /// cuMemsetD8Async (stream-ordered, graph-safe) y reduce atomicAdd.
    pub fn q3kGemmM1Dp4a(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 256 == 0);
        // S auto INTRA: n≤1024 ⇒ 4 (attn_k/v: 2 filas/bloque × 4 partes),
        // n≤3072 ⇒ 2 (attn_q/o: 4 filas/bloque × 2), mayor ⇒ 1 (saturado).
        // AU3_SPLIT: override manual para tuning/bench (0/1/2/4 = S usado).
        // Microbench 2026-09-10: split-K (inter E intra) NEGATIVO en n
        // pequeño (0.74-0.85×: fase-1 re-pagada, memset+atomicAdd) y el
        // S=1 puro ya empató al escalar ⇒ default 1; el ILP-2 del bucle
        // es la palanca activa (2 SBs/warp en vuelo, acumuladores par/impar).
        const au3s = std.c.getenv("AU3_SPLIT");
        const split: usize = if (au3s) |sv| (std.fmt.parseInt(usize, std.mem.span(sv), 10) catch 1) else 1;
        if (split > 1) try cudaz.cuMemsetD8Async(@intCast(out), 0, n * @sizeOf(f32), self.stream);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        var s1: c_int = -n_c(split); // negativo = intra-block
        const func = try self.get("q3kGemmM1Dp4aKernel");
        const kb = k / 32;
        const smem: c_uint = @intCast(((k + 15) & ~@as(usize, 15)) + kb * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1, &s1 };
        try cudaz.cuLaunchKernel(func, n_u((n * split + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// a-U3 fase 3: M=1 q3_k GEMV dp4a sobre REPACK alineado 128B/SB
    /// (experimento de hipótesis: la amplificación de sectores del stride
    /// 110B — byte-loads ld32_unaligned ~2.5× — es EL cuello; el layout
    /// [hm32|qs64|s16_16|d@112|pad] con SB múltiplo de 16 da loads
    /// coalesced puros y escalas pre-decodificadas en el pack). `b` DEBE
    /// ser un buffer packed de N×(k/256)×128 bytes (ver test a-U3 repack:
    /// el pack lo hace el test en CPU; si la hipótesis se valida, sube a
    /// cache packed on-load estilo q4PackedWeight). Requisito K%256==0.
    pub fn q3kGemmM1Dp4aPacked(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 256 == 0);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("q3kGemmM1Dp4aPackedKernel");
        const kb = k / 32;
        const smem: c_uint = @intCast(((k + 15) & ~@as(usize, 15)) + kb * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// a-U3 fase 3c: fusión QKV — UN launch para las 3 proyecciones q/k/v
    /// con la MISMA A (x del token). El camino separado paga 3 launches +
    /// 3 fase-1 (A→q8 idéntica ×3) por capa; el fused paga 1 y 1. Los
    /// pesos bq/bk/bv deben ser packed 128B/SB (q3kPackedWeight, MISMO K
    /// y qtype q3_k); outs siguen siendo los buffers separados del attn
    /// (cero cambios en consumers). Requisito: K % 256 == 0.
    pub fn q3kGemmM1Dp4aPackedQKV(self: *LayerKernels, a: usize, bq: usize, bk: usize, bv: usize, cq: usize, ck: usize, cv: usize, k: usize, n_q: usize, n_k: usize, n_v: usize) !void {
        std.debug.assert(k % 256 == 0);
        var av = a;
        var bqv = bq;
        var bkv = bk;
        var bvv = bv;
        var cqv = cq;
        var ckv = ck;
        var cvv = cv;
        var k1: c_int = n_c(k);
        var nq1: c_int = n_c(n_q);
        var nk1: c_int = n_c(n_k);
        var nv1: c_int = n_c(n_v);
        const func = try self.get("q3kGemmM1Dp4aPackedQKVKernel");
        const kb = k / 32;
        const smem: c_uint = @intCast(((k + 15) & ~@as(usize, 15)) + kb * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bqv, &bkv, &bvv, &cqv, &ckv, &cvv, &k1, &nq1, &nk1, &nv1 };
        const total = n_q + n_k + n_v;
        try cudaz.cuLaunchKernel(func, n_u((total + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// STUDY §5.8: M=1 q4_0 GEMV dp4a con cuantización de A FUSIONADA — un
    /// solo lanzamiento (vs 2-launch del MMQ o el FMA-escalar del clásico).
    /// Perfil nsys: los GEMVs qgemm son el 68% del decode a ~26% HBM; este
    /// kernel ataca el ancho de banda (funnel-shift 18B + 8 dp4a/bloque)
    /// sin pagar lanzamientos extra. Requisito: K % 32 == 0. Grid ceil(N/8),
    /// block 256; shared: aq pad16 | d f32 | sa f32.
    pub fn q4gemmM1Dp4a(self: *LayerKernels, a: usize, b: usize, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 32 == 0);
        var av = a;
        var bv = b;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("q4gemmM1Dp4aKernel");
        const kb_total = k / 32;
        const aq_pad = (k + 15) & ~@as(usize, 15);
        const smem: c_uint = @intCast(aq_pad + 2 * kb_total * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// 1.3 (lane-f): M1 sobre peso REPACKED dual-view (uint4 payload + d).
    /// `w_bytes` es el puntero HOST original (key del cache packed); el
    /// dispatcher resuelve payload/d en device. Gate: env Q4PACK=1 (opt-in).
    pub fn q4gemmM1Dp4aPacked(self: *LayerKernels, allocator: std.mem.Allocator, a: usize, w_bytes: []const u8, out: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 32 == 0);
        const pk = try q4PackedWeight(allocator, @intFromPtr(w_bytes.ptr), w_bytes, n, k);
        var av = a;
        var bpv: usize = pk.payload;
        var bdv: usize = pk.d;
        var ov = out;
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("q4gemmM1Dp4aPackedKernel");
        const kb_total = k / 32;
        const aq_pad = (k + 15) & ~@as(usize, 15);
        const smem: c_uint = @intCast(aq_pad + 2 * kb_total * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bpv, &bdv, &ov, &k1, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), 1, 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// STUDY §5.9: GEMM q4_0 M≤32 dp4a — prefill tier. Un lanzamiento por
    /// trozo de ≤32 tokens (m>32 → loop del caller: 512 tok = 16 launches).
    /// El peso se lee de HBM UNA vez por trozo (el clásico lo re-leía m×).
    /// Shared (M=32): aq pad16 [M*K] + d/sa f32 [M*KB] — 80KB @K=2048, 40KB
    /// @K=1024 (≤99KB opt-in sm_86). M_real = m_chunk (troza el dispatch).
    pub fn q4gemmMDp4a(self: *LayerKernels, a: usize, b: usize, out: usize, m: usize, k: usize, n: usize) !void {
        std.debug.assert(k % 32 == 0);
        std.debug.assert(m >= 1 and m <= 32);
        var av = a;
        var bv = b;
        var ov = out;
        var m1: c_int = n_c(m);
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        const func = try self.get("q4gemmMDp4aKernel");
        // v3: grid clásico ((n+7)/8, m); block 256. Shared: 1 token (K i8
        // + 2·KB f32). Tráfico de peso = clásico; inner ~2-4× por dp4a.
        const kb_total = k / 32;
        const aq_pad = (k + 15) & ~@as(usize, 15);
        const smem: c_uint = @intCast(aq_pad + 2 * kb_total * @sizeOf(f32));
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &k1, &n1, &m1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), n_u(m), 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
    }

    /// Proyección Q4_0 M=1 con peso cuantizado: sube los bytes Q4_0 una sola vez
    /// (cache módulo) y lanza `q4gemmM1`. `w_bytes` = tensor GGUF [in,out] Q4_0.
    pub fn q4gemmLinear(self: *LayerKernels, allocator: std.mem.Allocator, x: usize, w_bytes: []const u8, out: usize, k: usize, n: usize) !void {
        const dev = try q4Weight(allocator, @intFromPtr(w_bytes.ptr), w_bytes);
        // STUDY §5.4: lm_head MMQ split-K (lane-b B3, dos pasos: quantize-A →
        // GEMV q4_0). Para GEMVs anchos (n=248320 ⇒ grid (n+7)/8 × K-entero
        // por warp con lecturas escalares) la variante MMQ (dp4a + split-K
        // gridDim.z=4 + funnel-shift en el stride 18B) duplica el ancho de
        // banda efectivo del peso 127MB/token. OPT-IN: LMSPLIT=1 — unset =
        // q4gemmM1 clásico (árbol nunca decode-unsafe con WIP experimental).
        // Buffers de cuantización de A persistentes (M=1: 4KB + 128B).
        if (std.c.getenv("LMSPLIT") != null and k % 32 == 0) {
            const aq_pad = (1 * k + 15) & ~@as(usize, 15);
            const kb = k / 32;
            // aq i8 pad16 | ad __half [1*KB] | asa int [1*KB]
            try self.ensureMmqABuf(aq_pad + kb * 2 + kb * 4);
            const aq = self.mmq_a_buf;
            const ad = aq + aq_pad;
            const asa = ad + kb * @sizeOf(u16);
            debugz.dbg.printLevel(.detail, "[lm_splitk] n={d} k={d} → mmq Q4_0 (aq=0x{x})\n", .{ n, k, aq });
            try self.mmqQuantizeA(x, aq, ad, asa, 1, k);
            try cudaz.cuMemsetD8(out, 0, 1 * n * @sizeOf(f32));
            try self.mmqQ4_0GEMV(aq, ad, asa, dev, out, 1, k, n);
            return;
        }
        // STUDY §5.8: GEMV dp4a de UN lanzamiento (quantize-A fused). Perfil
        // nsys: los qgemm M=1 son el 68% del decode a ~26% HBM — el dp4a +
        // funnel-shift los lleva al ancho de banda; a diferencia del LMSPLIT
        // (2-launch + memset, sólo gana en N enorme) aquí no hay overhead de
        // lanzamiento extra. Gate N ≥ 2048 (misma razón que el hook M>1: el
        // coste fijo de cuantizar A no se amortiza en N pequeño).
        // G0 (TODO 1.9, lane-c): DEFAULT-ON — medido 75.9→150.4 t/s (+98%
        // usuarios default, A/B trunk 2026-09-08), paridad e2e byte-idéntica
        // demostrada @ee75066. Opt-out NOSSM4DP4A=1 (A/B del canónico).
        if (!debugz.dbg.no_ssm4dp4a and k % 32 == 0 and n >= 2048) {
            debugz.dbg.printLevel(.detail, "[§5.8-dp4a] n={d} k={d}\n", .{ n, k });
            try self.q4gemmM1Dp4a(x, dev, out, k, n);
            return;
        }
        try self.q4gemmM1(x, dev, out, k, n);
    }

    /// GEMM cuantizado batched: C[M,N] = A[M,K] * B_q[K,N]. `qtype`:
    /// 0=q4_0, 1=q4_1, 2=q5_k, 3=q6_k, 4=q4_k, 5=q8_0, 6=q3_k, 7=q2_k,
    /// 8=iq3_s, 9=iq2_s, 10=iq4_nl, 11=mxfp4, 12=iq3_xxs, 13=iq2_xxs,
    /// 14=iq2_xs, 15=tq2_0, 16=iq1_m. Pesos GGUF [in,out] sin dequantizar.
    /// [in,out] sin dequantizar.
    pub fn qgemmLinear(self: *LayerKernels, allocator: std.mem.Allocator, a: usize, w_bytes: []const u8, out: usize, m: usize, k: usize, n: usize, qtype: u32) !void {
        debugz.dbg.printLevel(.detail, "[qgemmLinear] m={d} n={d} k={d} qtype={d}\n", .{ m, n, k, qtype });
        const dev = try q4Weight(allocator, @intFromPtr(w_bytes.ptr), w_bytes);
        // STUDY §5.8/§5.9: q4_0 por dp4a — M=1 (decode, un launch) y M>1
        // (prefill). PERFIL REAL (nsys, prompt 15 tok — el alias -p roto
        // hacía que todos los benches previos corrieran con "Hola" 1-tok):
        // dp4a GANA en N grande (qkv N=6144: 380→273µs; z N=2048:
        // 137→100µs; ffn N=3584: 228→165µs) y PIERDE en N pequeño (out
        // N=1024: 230→273µs) — el coste fijo de la fase-1 de cuantización
        // no se amortiza con pocas filas. Gate por tamaño: sólo N ≥ 2048.
        // G0 (TODO 1.9, lane-c): DEFAULT-ON (ver q4gemmLinear). Opt-out
        // NOSSM4DP4A=1.
        if (qtype == 0 and !debugz.dbg.no_ssm4dp4a and k % 32 == 0 and n >= 2048) {
            // 1.3 (lane-f): Q4PACK=1 → M1 usa el peso repackeado dual-view.
            // PRE-WARM aquí (path prefill, m>1, FUERA de CUDA-graph capture):
            // el repack hace cuMemAlloc+H2D lazy — ilegal dentro de capture,
            // desactivaba el graph en silencio (177→90 t/s). Con el warm el
            // decode siempre cache-hit → launch puro → capture-safe.
            if (std.c.getenv("Q4PACK") != null) {
                _ = q4PackedWeight(allocator, @intFromPtr(w_bytes.ptr), w_bytes, n, k) catch {};
            }
            if (m == 1) {
                // 1.3 (lane-f): con Q4PACK=1 el M1 consume el peso repackeado
                // dual-view (uint4 payload) — menos transacciones HBM. Los
                // pesos viven en cache packed PARALELO (evict junto al canónico).
                if (std.c.getenv("Q4PACK") != null) {
                    debugz.dbg.printLevel(.detail, "[1.3-packed] qgemmLinear n={d} k={d}\n", .{ n, k });
                    try self.q4gemmM1Dp4aPacked(allocator, a, w_bytes, out, k, n);
                    return;
                }
                debugz.dbg.printLevel(.detail, "[§5.8-dp4a] qgemmLinear n={d} k={d}\n", .{ n, k });
                try self.q4gemmM1Dp4a(a, dev, out, k, n);
                return;
            }
            // §5.10 (lane-f): WMMA tensor-core GEMM for M≥128. Single launch
            // instead of M/32 dp4a launches. Dequant q4_0→int8 in shared mem.
            if (m >= 128 and std.c.getenv("NOWMA") == null) {
                debugz.dbg.printLevel(.detail, "[§5.10-wmma] qgemmLinear m={d} n={d} k={d}\n", .{ m, n, k });
                const kb = k / 32;
                const aq_pad = (m * k + 15) & ~@as(usize, 15);
                // aq i8 [M,K] pad16 | ad __half [M*KB] | asa int [M*KB]
                const ad_bytes = m * kb * @sizeOf(u16);
                const asa_bytes = m * kb * @sizeOf(i32);
                try self.ensureMmqABuf(aq_pad + ad_bytes + asa_bytes);
                const aq = self.mmq_a_buf;
                const ad = aq + aq_pad;
                const asa = ad + ad_bytes;
                try self.mmqQuantizeA(a, aq, ad, asa, m, k);
                cudaz.cuStreamSynchronize(self.stream) catch |e| {
                    debugz.dbg.print("[§5.10-wmma] FALLO sync post-quantize: {s}\n", .{@errorName(e)});
                    return e;
                };
                try self.mmqQ4_0GEMM(aq, ad, asa, dev, out, m, k, n);
                cudaz.cuStreamSynchronize(self.stream) catch |e| {
                    debugz.dbg.print("[§5.10-wmma] FALLO sync post-gemm: {s}\n", .{@errorName(e)});
                    return e;
                };
                return;
            }
            // §5.9: trocear m en chunks de ≤32 (ubatch 512 → 16 launches).
            var m_off: usize = 0;
            while (m_off < m) {
                const m_chunk: usize = @min(32, m - m_off);
                const a_chunk = a + m_off * k * @sizeOf(f32);
                const c_chunk = out + m_off * n * @sizeOf(f32);
                try self.q4gemmMDp4a(a_chunk, dev, c_chunk, m_chunk, k, n);
                m_off += m_chunk;
            }
            debugz.dbg.printLevel(.detail, "[§5.9-dp4a] qgemmLinear m={d} n={d} k={d} ({d} launches)\n", .{ m, n, k, (m + 31) / 32 });
            return;
        }
        // STUDY 1.1: q5_k M=1 (ssm_out) — smem compartido de A entre filas
        // (1.58× vs canónico, paridad bit-exact). Requisito: K % 256 == 0
        // (SB Q5_K). M>1 queda en qgemm clásico (prefill ssm_out es raro y
        // el troceo M≤32 de §5.9 no aplica a SB-256 sin adaptar el kernel).
        // G0 (TODO 1.9, lane-c): DEFAULT-ON (familia SSM4DP4A, ver
        // q4gemmLinear). Opt-out NOSSM4DP4A=1.
        if (qtype == 2 and m == 1 and k % 256 == 0 and !debugz.dbg.no_ssm4dp4a) {
            debugz.dbg.printLevel(.detail, "[1.1-q5] qgemmLinear n={d} k={d}\n", .{ n, k });
            try self.q5gemmM1(a, dev, out, k, n);
            return;
        }
        // STUDY 1.2: q6_k M=1 (lm_head 248320×1024 del 0.8B + FFN-down) —
        // smem compartido de A entre filas (mismo win que 1.1; espejo de
        // lane_q6k_val). OJO: N puede ser 248320 → el lm_head del decode
        // es EL GEMV dominante del token (~860µs en el perfil real).
        // G0 (TODO 1.9, lane-c): DEFAULT-ON (familia SSM4DP4A, ver
        // q4gemmLinear). Opt-out NOSSM4DP4A=1.
        if (qtype == 3 and m == 1 and k % 256 == 0 and !debugz.dbg.no_ssm4dp4a) {
            debugz.dbg.printLevel(.detail, "[1.2-q6] qgemmLinear n={d} k={d}\n", .{ n, k });
            try self.q6gemmM1(a, dev, out, k, n);
            return;
        }
        // U3 (lane-a, eje §12): q4_k M=1 — EL hueco del Q4_K_XL (97% del
        // token era SSM con proyecciones q4_k por el GENÉRICO; 50 vs 153
        // t/s del Q5_K_S todo-especializado). Espejo del case 4 del
        // qgemmKernel; mismo gate que 1.1/1.2 (familia SSM4DP4A, opt-out
        // NOSSM4DP4A=1). Q3_K_S del gate 3B es q4_k dominante ⇒ de-riesga
        // el ≥60 t/s cuando U1 aterrice.
        if (qtype == 4 and m == 1 and k % 256 == 0 and !debugz.dbg.no_ssm4dp4a) {
            debugz.dbg.printLevel(.detail, "[12-q4k] qgemmLinear n={d} k={d}\n", .{ n, k });
            try self.q4kGemmM1(a, dev, out, k, n);
            return;
        }
        // U3: q8_0 M=1 — 2º hueco (bancos ssm_* ΔNet). Espejo del case 5.
        if (qtype == 5 and m == 1 and k % 32 == 0 and !debugz.dbg.no_ssm4dp4a) {
            debugz.dbg.printLevel(.detail, "[12-q8k] qgemmLinear n={d} k={d}\n", .{ n, k });
            try self.q8kGemmM1(a, dev, out, k, n);
            return;
        }
        // a-U3: q3_k M=1 dp4a — EL cuello del gate 3B (Q3_K_S todo q3_k:
        // 712µs/capa escalar = compute-bound, 16% techo HBM). Formulación
        // llama.cpp vec_dot_q3_K_q8_1 sobre layout 110B; familia SSM4DP4A.
        // Fase 3b: DEFAULT ON (opt-out NOQ3PACK=1) → M=1 consume el peso REPACKED 128B/SB
        // (coalesced, 1.52× suma/capa medido — el stride 110B amplificaba
        // sectores ~2.5×). PRE-WARM en el path m>1 (prefill, FUERA de
        // CUDA-graph capture — patrón 1.3 lane-f: cuMemAlloc+H2D lazy es
        // ilegal en capture); decode siempre cache-hit → launch puro.
        if (qtype == 6 and m == 1 and k % 256 == 0 and !debugz.dbg.no_ssm4dp4a) {
            if (std.c.getenv("NOQ3PACK") == null) {
                const pkdev = q3kPackedWeight(allocator, @intFromPtr(w_bytes.ptr), w_bytes, n, k) catch null;
                if (pkdev) |pd| {
                    debugz.dbg.printLevel(.detail, "[aU3-q3pack] qgemmLinear n={d} k={d}\n", .{ n, k });
                    try self.q3kGemmM1Dp4aPacked(a, pd, out, k, n);
                    return;
                }
            }
            debugz.dbg.printLevel(.detail, "[aU3-q3k] qgemmLinear n={d} k={d}\n", .{ n, k });
            try self.q3kGemmM1Dp4a(a, dev, out, k, n);
            return;
        }
        // a-U3 fase 3b: prewarm del packed en el path m>1 (prefill, default on; NOQ3PACK=1 lo desactiva) —
        // el hook de arriba solo ve m==1. catch {}: si el pack falla
        // (OOM etc.) el decode cae al 110B sin graph-roto.
        if (qtype == 6 and m > 1 and k % 256 == 0 and std.c.getenv("NOQ3PACK") == null) {
            _ = q3kPackedWeight(allocator, @intFromPtr(w_bytes.ptr), w_bytes, n, k) catch {};
        }
        // P0-5/P0-6: exotic qtypes M=1 dp4a — iq3_s, iq2_s, iq4_xs, q4_1, q2_k, iq4_nl, iq3_xxs, iq2_xxs, iq2_xs
        if (m == 1 and k % 256 == 0 and !debugz.dbg.no_ssm4dp4a) {
            switch (qtype) {
                3 => { // q4_1
                    debugz.dbg.printLevel(.detail, "[P0-6-q4_1] qgemmLinear n={d} k={d}\n", .{ n, k });
                    try self.q41GemmM1Dp4a(a, dev, out, k, n);
                    return;
                },
                10 => { // q2_k
                    debugz.dbg.printLevel(.detail, "[P0-6-q2_k] qgemmLinear n={d} k={d}\n", .{ n, k });
                    try self.q2kGemmM1Dp4a(a, dev, out, k, n);
                    return;
                },
                16 => { // iq2_xxs
                    debugz.dbg.printLevel(.detail, "[P0-6-iq2_xxs] qgemmLinear n={d} k={d}\n", .{ n, k });
                    try self.iq2xxsGemmM1Dp4a(a, dev, out, k, n);
                    return;
                },
                17 => { // iq2_xs
                    debugz.dbg.printLevel(.detail, "[P0-6-iq2_xs] qgemmLinear n={d} k={d}\n", .{ n, k });
                    try self.iq2xsGemmM1Dp4a(a, dev, out, k, n);
                    return;
                },
                18 => { // iq3_xxs
                    debugz.dbg.printLevel(.detail, "[P0-6-iq3_xxs] qgemmLinear n={d} k={d}\n", .{ n, k });
                    try self.iq3xxsGemmM1Dp4a(a, dev, out, k, n);
                    return;
                },
                20 => { // iq4_nl
                    debugz.dbg.printLevel(.detail, "[P0-6-iq4_nl] qgemmLinear n={d} k={d}\n", .{ n, k });
                    try self.iq4nlGemmM1Dp4a(a, dev, out, k, n);
                    return;
                },
                21 => { // iq3_s
                    debugz.dbg.printLevel(.detail, "[P0-5-iq3_s] qgemmLinear n={d} k={d}\n", .{ n, k });
                    try self.iq3sGemmM1Dp4a(a, dev, out, k, n);
                    return;
                },
                22 => { // iq2_s
                    debugz.dbg.printLevel(.detail, "[P0-5-iq2_s] qgemmLinear n={d} k={d}\n", .{ n, k });
                    try self.iq2sGemmM1Dp4a(a, dev, out, k, n);
                    return;
                },
                23 => { // iq4_xs
                    debugz.dbg.printLevel(.detail, "[P0-5-iq4_xs] qgemmLinear n={d} k={d}\n", .{ n, k });
                    try self.iq4xsGemmM1Dp4a(a, dev, out, k, n);
                    return;
                },
                else => {},
            }
        }
        try self.qgemm(a, dev, out, m, k, n, qtype);
    }

    pub fn qgemm(self: *LayerKernels, a: usize, b: usize, out: usize, m: usize, k: usize, n: usize, qtype: u32) !void {
        var av = a;
        var bv = b;
        var ov = out;
        var m1: c_int = n_c(m);
        var k1: c_int = n_c(k);
        var n1: c_int = n_c(n);
        var t1: c_int = @intCast(qtype);
        const func = try getT(.qgemmKernel); // UC-3.2: typo = compile error
        const smem: c_uint = @intCast(k * @sizeOf(f32));
        // T1 VRAM-spec (lane-a): k grande — p.ej. ffn_down con
        // intermediate_dim=17408 ⇒ smem=68KB > 48KB por defecto. Sin opt-in el
        // launch falla (CUDA_ERROR_INVALID_VALUE en sm_86, límite 99KB/bloque).
        // 8 = CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES. Idempotente.
        cudaz.cuFuncSetAttribute(func, 8, 99 * 1024) catch {};
        // lane-cuda UC-2.3 (piloto ErrorFlag): buffer persistente PRE-alloc
        // (capture-safe, lección 1.3 — nunca alloc dentro de capture). El
        // clear es async (stream-ordered) = legal en graph capture. El CHECK
        // post-launch (D2H síncrono) es OPT-IN via ZIG_AI_EF_CHECK porque
        // rompería el capture del decode — off por defecto, cero coste.
        if (self.ef_buf == 0) {
            self.ef_buf = try cudaz.cuMemAlloc(4);
            if (debugz.dbg.at(.info)) debugz.dbg.printLevel(.info, "[gpu_kernels] ErrorFlag buf alloc (qgemm piloto UC-2.3)\n", .{});
        }
        try cudaz.cuMemsetD8Async(self.ef_buf, 0, 4, self.stream);
        var efv: usize = self.ef_buf;
        var kp = [_]?*anyopaque{ &av, &bv, &ov, &m1, &k1, &n1, &t1, &efv };
        try cudaz.cuLaunchKernel(func, n_u((n + 7) / 8), n_u(m), 1, 256, 1, 1, smem, self.stream, @ptrCast(&kp), null);
        // dbg-700: sync tras cada launch (bisect de carrera async). QGEMV_SYNC=1
        if (std.c.getenv("QGEMV_SYNC") != null) {
            cudaz.cuStreamSynchronize(self.stream) catch {};
        }
        // lane-cuda UC-2.3: check OPT-IN (ZIG_AI_EF_CHECK=1) — sincroniza y
        // lee el flag (4B D2H). Fuera de CUDA-graph runs (bench/tests/dumps).
        if (std.c.getenv("ZIG_AI_EF_CHECK") != null) {
            cudaz.cuStreamSynchronize(self.stream) catch {};
            var code: u32 = 0;
            cudaz.cuMemcpyDtoH(@intFromPtr(&code), self.ef_buf, 4) catch {};
            if (code != 0) {
                const name: []const u8 = switch (code) { // espejo error_flag.zig Code
                    1 => "OOB",
                    2 => "NAN",
                    3 => "INF",
                    4 => "ASSERT",
                    else => "CUSTOM/UNKNOWN",
                };
                debugz.dbg.printLevel(.info, "[gpu_kernels] qgemm ErrorFlag: {s} ({d}) — M={d} K={d} N={d} qtype={d}\n", .{ name, code, m, k, n, qtype });
            }
        }
    }

    // ─── DFlash2 selector top-K (5.3, lane-cuda) ─────────────────────────────────
    //
    // Wrapper del kernel `dflash2TopKKernel` (layer_kernels.cu). Selecciona
    // top-K expertos sobre un array de scores device.
    //
    // Parámetros:
    //   scores  — deviceptr a scores [num_experts] f32
    //   ids_out — deviceptr a ids   [top_k] i32
    //   vals_out— deviceptr a vals  [top_k] f32 (softmax-normalizados)
    //   num_experts, top_k — escalares host
    //
    // Grid/Block: grid=(1,1), block=(min(num_experts,1024),1,1), smem=num_experts*4B.
    // Gate: requiere ZIG_AI_DFLASH2_TOPK_GPU=1; sin él, usa CPU naive.
    pub fn dflash2TopK(
        self: *LayerKernels,
        scores: usize,
        ids_out: usize,
        vals_out: usize,
        num_experts: usize,
        top_k: usize,
    ) !void {
        const gpu_enabled = std.c.getenv("ZIG_AI_DFLASH2_TOPK_GPU") != null;
        if (!gpu_enabled) return error.NotImplemented;
        if (num_experts > 1024) return error.InvalidArgument;
        const func = try self.get("dflash2TopKKernel");
        const n_i32: i32 = @intCast(@min(num_experts, 1024));
        const k_i32: i32 = @intCast(top_k);
        var kp = [_]?*anyopaque{ @constCast(&scores), @constCast(&ids_out), @constCast(&vals_out), @constCast(&n_i32), @constCast(&k_i32) };
        try cudaz.cuLaunchKernel(
            func,
            1, 1, 1,                     // grid
            @intCast(@min(num_experts, 1024)), 1, 1, // block
            @intCast(num_experts * @sizeOf(f32)), // smem
            self.stream,
            @ptrCast(&kp),
            null,
        );
    }

    /// Tree-walk: selección top-K sobre árbol binario de scores.
    /// Usa `dflash2TreeWalkKernel` cuando `selector_rank < num_experts`.
    pub fn dflash2TreeWalk(
        self: *LayerKernels,
        scores: usize,
        ids_out: usize,
        vals_out: usize,
        num_experts: usize,
        top_k: usize,
    ) !void {
        const gpu_enabled = std.c.getenv("ZIG_AI_DFLASH2_TOPK_GPU") != null;
        if (!gpu_enabled) return error.NotImplemented;
        if (num_experts > 1024) return error.InvalidArgument;
        const func = try self.get("dflash2TreeWalkKernel");
        const n_i32: i32 = @intCast(@min(num_experts, 1024));
        const k_i32: i32 = @intCast(top_k);
        var kp = [_]?*anyopaque{ @constCast(&scores), @constCast(&ids_out), @constCast(&vals_out), @constCast(&n_i32), @constCast(&k_i32) };
        try cudaz.cuLaunchKernel(
            func,
            1, 1, 1,                     // grid
            @intCast(@min(num_experts, 1024)), 1, 1, // block
            @intCast(num_experts * @sizeOf(f32)), // smem
            self.stream,
            @ptrCast(&kp),
            null,
        );
    }
};

var q4_cache: ?std.AutoHashMap(usize, usize) = null;
/// 1.3 (lane-f): device ptr del peso q4_0 REPACKED (dual-view payload|d).
/// Key = mismo host ptr que q4_cache; value = payload (el array d viaja
/// en un alloc contiguo justo DESPUÉS — ver q4PackedWeight).
var q4_packed_cache: ?std.AutoHashMap(usize, usize) = null;
/// a-U3 fase 3b (lane-a): device ptr del peso q3_k REPACKED 128B/SB
/// ([hm32|qs64|s16_16|d@112|pad14], escalas pre-decodificadas). Key =
/// mismo host ptr que q4_cache — comparte vida/eviction (1.52× GEMV,
/// ver q3kGemmM1Dp4aPackedKernel).
var q3_packed_cache: ?std.AutoHashMap(usize, usize) = null;
var q3_packed_cache_bytes: usize = 0;
/// Caché NEGATIVA del repack q3_k: si repack/cuMemAlloc falla (p.ej. VRAM
/// insuficiente), no reintentar en cada token (host repack + alloc fallida
/// por token = thrash). El caller cae al kernel 110B y queda así.
var q3_pack_failed: bool = false;
var q4_packed_cache_bytes: usize = 0;
var q4_cache_bytes: usize = 0;
/// T1 VRAM-spec (lane-a): techo del cache de pesos cuantizados en device.
/// Con layer-streaming el cache acumulaba el modelo COMPLETO (~31GB en
/// Q8_K_XL) y mataba la VRAM igual que weight_cache antes de los hooks de D
/// (CudaMallocFailed en q4Weight, e2e_27b_v3). Al superar el techo se libera
/// TODO: cuMemFree es diferido-seguro (el driver no libera hasta que ningún
/// kernel pendiente use el buffer), así que limpiar entre launches es
/// correcto; el coste es re-subir el peso cuantizado en el próximo uso
/// (~15-45ms/capa por PCIe, aceptable v1). 0 = sin techo (comportamiento
/// histórico). Env: ZIG_AI_Q4CACHE_MAX_MB.
var q4_cache_max_bytes: ?usize = null;

fn q4CacheMaxBytes() usize {
    if (q4_cache_max_bytes) |v| return v;
    // Default 0 = sin techo fijo: la expulsión la gobierna la VRAM libre
    // (ver q4Weight). El env queda como override duro para experimentos.
    const mb: usize = blk: {
        const s = std.c.getenv("ZIG_AI_Q4CACHE_MAX_MB") orelse break :blk 0;
        break :blk std.fmt.parseInt(usize, std.mem.span(s), 10) catch 0;
    };
    const v = mb * 1024 * 1024;
    q4_cache_max_bytes = v;
    return v;
}

pub fn evictQ4CacheAll() void {
    if (q4_cache) |*m| {
        var it = m.iterator();
        while (it.next()) |entry| {
            cudaz.cuMemFree(@intCast(entry.value_ptr.*));
        }
        m.deinit();
        q4_cache = null;
    }
    q4_cache_bytes = 0;
    // 1.3 (lane-f): el cache packed comparte vida con el canónico.
    if (q4_packed_cache) |*m| {
        var it = m.iterator();
        while (it.next()) |entry| {
            cudaz.cuMemFree(@intCast(entry.value_ptr.*));
        }
        m.deinit();
        q4_packed_cache = null;
    }
    // a-U3 fase 3b (lane-a): el q3_k packed ídem — misma vida que el
    // canónico (evict conjunto, sin dobles frees: caches disjuntos).
    if (q3_packed_cache) |*m| {
        var it = m.iterator();
        while (it.next()) |entry| {
            cudaz.cuMemFree(@intCast(entry.value_ptr.*));
        }
        m.deinit();
        q3_packed_cache = null;
    }
    q3_packed_cache_bytes = 0;
}

pub fn deinitQ4Cache() void {
    evictQ4CacheAll();
}

// ── 1.3 (lane-f): pre-repack q4_0 → layout dual [payload 16B][d f16] ──────
// El q4_0 canónico es un SB de 18B ([d f16][16B quanta]) — el GEMV dp4a lo
// lee con funnel-shift (5 loads u32 desalineados por lane). El repack
// separa payload (16B, uint4-aligned, contiguos por fila) y escalas d
// (array f16 propio): el lane lee 1×uint4 limpio por KB. Mismo contenido,
// distinta dirección — bit-exact por construcción.
//
// Entrada: bytes canónicos de UNA matriz [N filas][K cols] q4_0
//          (kb_total = K/32 bloques por fila, fila stride kb_total*18).
// Salida: payload []u8 de N*kb_total*16 y d []f16 de N*kb_total.
pub fn repackQ4PayloadD(bytes: []const u8, n_rows: usize, k: usize, allocator: std.mem.Allocator) !struct { payload: []u8, d: []f16 } {
    const kb_total = k / 32;
    if (kb_total == 0 or n_rows == 0) return error.ShapeUnsupported;
    if (bytes.len < n_rows * kb_total * 18) return error.BufferTooSmall;
    const payload = try allocator.alloc(u8, n_rows * kb_total * 16);
    errdefer allocator.free(payload);
    const d = try allocator.alloc(f16, n_rows * kb_total);
    errdefer allocator.free(d);
    for (0..n_rows) |r| {
        const row_src = bytes[r * kb_total * 18 ..];
        const row_pay = payload[r * kb_total * 16 ..];
        const row_d = d[r * kb_total ..];
        for (0..kb_total) |kb| {
            const sb = row_src[kb * 18 ..][0..18];
            row_d[kb] = @bitCast(sb[0] | (@as(u16, sb[1]) << 8));
            @memcpy(row_pay[kb * 16 ..][0..16], sb[2..18]);
        }
    }
    return .{ .payload = payload, .d = d };
}

/// 1.3 (lane-f): sube (o reutiliza) la vista packed de un peso q4_0.
/// Los bytes canónicos se repackean UNA vez a dual-view y viven en
/// q4_packed_cache. Devuelve {payload_dev, d_dev}. El d array se aloca
/// CONTIGUO tras el payload (un solo cuMemAlloc: [payload N*kb*16][d N*kb*2]).
pub fn q4PackedWeight(allocator: std.mem.Allocator, key: usize, bytes: []const u8, n_rows: usize, k: usize) !struct { payload: usize, d: usize } {
    if (q4_packed_cache) |*m| {
        if (m.get(key)) |dptr| {
            const kb_total = k / 32;
            return .{ .payload = dptr, .d = dptr + n_rows * kb_total * 16 };
        }
    }
    const rep = try repackQ4PayloadD(bytes, n_rows, k, allocator);
    defer allocator.free(rep.payload);
    defer allocator.free(rep.d);
    const total_b = rep.payload.len + rep.d.len * 2;
    const dev = try cudaz.cuMemAlloc(total_b);
    // Dos H2D separadas (payload, luego d) — evita problemas con
    // transferencias >2MB en drivers modeles/móviles.
    cudaz.cuMemcpyHtoD(dev, @intFromPtr(rep.payload.ptr), rep.payload.len) catch |e| {
        std.debug.print("[1.3] cuMemcpyHtoD payload failed: {any} (dev={x} len={d})\n", .{ e, dev, rep.payload.len });
        return e;
    };
    const d_bytes = std.mem.sliceAsBytes(rep.d);
    cudaz.cuMemcpyHtoD(dev + rep.payload.len, @intFromPtr(d_bytes.ptr), d_bytes.len) catch |e| {
        std.debug.print("[1.3] cuMemcpyHtoD d failed: {any} (dev={x} len={d})\n", .{ e, dev + rep.payload.len, d_bytes.len });
        return e;
    };
    if (q4_packed_cache == null) {
        q4_packed_cache = std.AutoHashMap(usize, usize).init(allocator);
    }
    try q4_packed_cache.?.put(key, dev);
    q4_packed_cache_bytes += total_b;
    return .{ .payload = dev, .d = dev + rep.payload.len };
}

// ── a-U3 fase 3b (lane-a): pre-repack q3_k → 128B/SB coalesced ─────────────
// El q3_k canónico es 110B/SB (stride no múltiplo de 4 ⇒ ld32_unaligned
// byte-loads, ~2.5× amplificación de sectores — el cuello real del GEMV,
// medido fase 2). El repack reordena [hm32|qs64|s16_16|d@112|pad14] con
// las escalas PRE-DECODIFICADAS (kmask-spill una vez, idéntico
// dequantQ3_K kv_quant.zig:2370): loads coalesced puros en el kernel.
// Mismo contenido numérico — paridad rel 1e-2 (drift 3e-6 medido).
// Entrada: bytes canónicos de UNA matriz [N filas][K cols] q3_k
//          (sb_total = K/256 SBs por fila, fila stride sb_total*110).
// Salida: []u8 de N*sb_total*128 (alineado 16 ⇒ uint4/ld.32 legales).
pub fn repackQ3K128(bytes: []const u8, n_rows: usize, k: usize, allocator: std.mem.Allocator) ![]u8 {
    const sb_total = k / 256;
    if (sb_total == 0 or n_rows == 0) return error.ShapeUnsupported;
    if (bytes.len < n_rows * sb_total * 110) return error.BufferTooSmall;
    const pk = try allocator.alloc(u8, n_rows * sb_total * 128);
    errdefer allocator.free(pk);
    @memset(pk, 0);
    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;
    for (0..n_rows) |r| {
        const row_src = bytes[r * sb_total * 110 ..];
        const row_dst = pk[r * sb_total * 128 ..];
        for (0..sb_total) |sb| {
            const bp = row_src[sb * 110 ..][0..110];
            const dp = row_dst[sb * 128 ..][0..128];
            @memcpy(dp[0..32], bp[0..32]); // hmask
            @memcpy(dp[32..96], bp[32..96]); // quanta 2-bit
            // kmask-spill: escalas crudas 12B → s16 16B pre-decodificadas.
            var aux: [4]u32 = @splat(0);
            @memcpy(@as([*]u8, @ptrCast(&aux))[0..12], bp[96..108]);
            const tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            @memcpy(dp[96..112], @as([*]const u8, @ptrCast(&aux))[0..16]);
            @memcpy(dp[112..114], bp[108..110]); // d f16
            // dp[114..128] pad 0 (memset inicial)
        }
    }
    return pk;
}

/// a-U3 fase 3b (lane-a): sube (o reutiliza) la vista packed 128B/SB de
/// un peso q3_k. Se repackea UNA vez a q3_packed_cache (key = host ptr,
/// igual que q4Weight/q4PackedWeight). VRAM: el modelo 3B Q3_K_S entero
/// cabe packed (~1.68GB vs 1.53 canónico, +10%).
pub fn q3kPackedWeight(allocator: std.mem.Allocator, key: usize, bytes: []const u8, n_rows: usize, k: usize) !usize {
    if (q3_pack_failed) return error.Q3PackDisabled;
    if (q3_packed_cache) |*m| {
        if (m.get(key)) |dptr| return dptr;
    }
    const pk = repackQ3K128(bytes, n_rows, k, allocator) catch |e| {
        q3_pack_failed = true;
        return e;
    };
    defer allocator.free(pk);
    const dev = cudaz.cuMemAlloc(pk.len) catch |e| {
        q3_pack_failed = true;
        return e;
    };
    try cudaz.cuMemcpyHtoD(dev, @intFromPtr(pk.ptr), pk.len);
    if (q3_packed_cache == null) {
        q3_packed_cache = std.AutoHashMap(usize, usize).init(allocator);
    }
    try q3_packed_cache.?.put(key, dev);
    q3_packed_cache_bytes += pk.len;
    debugz.dbg.printLevel(.info, "[aU3-q3pack] repack+upload n={d} k={d} {d}MB (cache {d}MB)\n", .{ n_rows, k, pk.len / (1024 * 1024), q3_packed_cache_bytes / (1024 * 1024) });
    return dev;
}

pub fn q4Weight(allocator: std.mem.Allocator, key: usize, bytes: []const u8) !usize {
    if (q4_cache) |*m| {
        if (m.get(key)) |d| {
            debugz.dbg.printLevel(.detail, "[matmul] q4Weight cache HIT key={x} dev={x} len={d}\n", .{ key, d, bytes.len });
            return d;
        }
    }
    debugz.dbg.printLevel(.detail, "[matmul] q4Weight MISS key={x} len={d}\n", .{ key, bytes.len });
    // T1 VRAM-spec v2 (lane-a): expulsar por VRAM LIBRE baja, no por techo
    // fijo — el techo 512MB hacía thrash a modelos residentes cuyo total
    // cuantizado lo superaba (LFM2.5-2.6B ≈1.7GB). Con umbral de margen,
    // un modelo residente nunca evicta y el streaming se autoregula.
    // cuMemFree es diferido-seguro bajo kernels en vuelo (semántica CUDA).
    const max_b = q4CacheMaxBytes();
    const hard_cap_hit = max_b > 0 and q4_cache_bytes + bytes.len > max_b;
    var free_b: usize = 0;
    var total_b: usize = 0;
    const low_vram = blk: {
        _ = cudaz.cuMemGetInfo(&free_b, &total_b) catch break :blk false;
        break :blk free_b < bytes.len + 512 * 1024 * 1024;
    };
    if (q4_cache != null and (hard_cap_hit or low_vram)) evictQ4CacheAll();
    const dev = try cudaz.cuMemAlloc(bytes.len);
    try cudaz.cuMemcpyHtoD(dev, @intFromPtr(bytes.ptr), bytes.len);
    if (q4_cache == null) {
        q4_cache = std.AutoHashMap(usize, usize).init(allocator);
    }
    // U3 (lane-a, eje §12): breadcrumb de la DECISIÓN de eviction — sin él
    // un churn por-token era indistinguible de un bug de routing. Muestra
    // el trigger real (cap duro vs VRAM libre) y los bytes en juego.
    if (q4_cache != null and (hard_cap_hit or low_vram)) {
        debugz.dbg.printLevel(.info, "[matmul] q4Weight EVICT trigger={} hard_cap_hit={} low_vram={} free={d}MB need={d}MB cached={d}MB\n", .{ hard_cap_hit or low_vram, hard_cap_hit, low_vram, free_b / (1024 * 1024), bytes.len / (1024 * 1024), q4_cache_bytes / (1024 * 1024) });
    }
    try q4_cache.?.put(key, dev);
    q4_cache_bytes += bytes.len;
    return dev;
}

