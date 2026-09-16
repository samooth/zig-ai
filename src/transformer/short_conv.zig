//! ShortConv Layer — LFM2 "short convolution" (depthwise conv1d + gating)
//! Similar to SSM conv but simpler: depthwise conv1d + silu -> in_proj (3x expand) ->
//! split(gate, up, value) -> silu(gate)*up + value -> out_proj
//!
//! Tensor names in GGUF (per layer):
//!   blk.{i}.shortconv.conv.weight       // [self.params.conv_dim, l_cache+1] depthwise conv1d kernel
//!   blk.{i}.shortconv.in_proj.weight    // [3 * self.params.conv_dim, n_embd] fused gate/up/value projection
//!   blk.{i}.shortconv.out_proj.weight   // [n_embd, self.params.conv_dim] output projection
//!   blk.{i}.attn_norm.weight            // pre-norm (shared with attention layers)
//!   blk.{i}.ffn_norm.weight             // post-conv norm (equivalent to post_attention_norm)
//!   blk.{i}.ffn_gate.weight             // FFN gate (shared with attention layers)
//!   blk.{i}.ffn_up.weight               // FFN up
//!   blk.{i}.ffn_down.weight             // FFN down
//!
//! No: post_attention_norm, attn_q_norm, attn_k_norm, attn_q/k/v/output
const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const cublas = @import("cublas");
const cudaz = @import("cudaz");
const layer_kernels = @import("layer_kernels");
const QuantWeight = @import("quant_weight").QuantWeight;
const gguf = @import("gguf");
const norm = @import("norm");
const debugz = @import("debug");
const SsmLayer = @import("ssm").SsmLayer;

pub const ShortConvError = error{
    WeightFileNotFound,
    ShapeMismatch,
};

pub const ShortConvParams = struct {
    n_embd: usize,
    conv_dim: usize, // = n_embd (LFM2 uses same dim)
    l_cache: usize, // kernel size - 1 (e.g., 3 for kernel=4)
    /// 8.3: kernel size REAL del GGUF (dims[0] del shortconv.conv.weight).
    /// En LFM2.5 es 3 (= l_cache); se setea en loadWeightsFromGguf.
    conv_kernel_gguf: usize = 0, // 0 = no cargado aún (usa l_cache+1 fallback)
    rms_eps: f32,

    pub fn convKernel(self: ShortConvParams) usize {
        // 8.3 FIX (lfm2.cpp ggml_ssm_conv:5562): d_conv = c->ne[0] — la
        // primera dim del GGUF shortconv.conv.weight (3, 2048) ES el kernel
        // size. La convención l_cache=kernel-1 venía de Qwen3.5-SSM (donde
        // l_cache=3, kernel=4 y el tensor tiene 4 filas). En LFM2.5 el
        // tensor tiene 3 filas = kernel 3; l_cache=3 (default GGUF) como
        // kernel-1 produciría kernel=4 leyendo un peso de más.
        if (self.conv_kernel_gguf > 0) return self.conv_kernel_gguf;
        return self.l_cache + 1;
    }

    /// 8.3 FIX: filas de estado recurrente. ggml_ssm_conv exige
    /// sx->ne[0] == d_conv - 1 + n_t (ggml.c:5562-5568) — el estado es
    /// kernel-1 filas, NO l_cache. Con kernel GGUF=3 el estado son 2 filas.
    /// El CPU forward concatenaba l_cache=3 filas (una de más) ⇒ off-by-one
    /// temporal: output[t] usaba state[2] donde ggml usa input[t-2].
    pub fn convStateLen(self: ShortConvParams) usize {
        return self.convKernel() - 1;
    }
};

/// 8.3 (LFM2.5): mapping dtype GGUF → código del qgemmKernel. ESPEJO de
/// SsmLayer.qgemmTypeFor (ssm no está cableado como import de short_conv y
/// añadir una dependencia cross-module por 1 función no compensa — misma
/// convención 0..16, mantener sincronizado).
fn qgemmTypeFor(t: gguf.GgmlType) ?u32 {
    return switch (t) {
        .q4_0 => 0,
        .q4_1 => 1,
        .q5_k => 2,
        .q6_k => 3,
        .q4_k => 4,
        .q8_0 => 5,
        .q3_k => 6,
        .q2_k => 7,
        .iq3_s => 8,
        .iq2_s => 9,
        .iq4_nl => 10,
        .mxfp4 => 11,
        .iq3_xxs => 12,
        .iq2_xxs => 13,
        .iq2_xs => 14,
        .tq2_0 => 15,
        .iq1_m => 16,
        // B-a4 (lane-a): case 17 = iq1_s (mapping GEMM 17/17).
        .iq1_s => 17,
        // Q1_0/Q2_0: cases 18/19 en qgemmKernel.
        .q1_0 => 18,
        .q2_0 => 19,
        else => null,
    };
}

pub const ShortConvLayer = struct {
    allocator: std.mem.Allocator,
    layer_idx: usize,
    params: ShortConvParams,
    matmul_engine: matmul.MatmulEngine,

    // Weights
    w_conv: QuantWeight, // [conv_dim, l_cache+1] depthwise conv1d
    w_in_proj: QuantWeight, // [3*conv_dim, n_embd] fused gate/up/value
    w_out_proj: QuantWeight, // [n_embd, conv_dim]

    // Norm weights (f32)
    attn_norm: Tensor(f32), // pre-norm [n_embd]
    ffn_norm: Tensor(f32), // post-conv norm [n_embd]

    // Scratch f32 (dequantized weights)
    scratch_conv: []f32, // conv_dim * (l_cache+1)
    scratch_in_proj: []f32, // (3*conv_dim) * n_embd
    scratch_out_proj: []f32, // n_embd * conv_dim

    // Recurrent state for decode: conv_state [l_cache, conv_dim]
    conv_state: []f32, // [l_cache * conv_dim]

    // GPU buffers
    gpu: ?ShortConvGpu = null,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        layer_idx: usize,
        params: ShortConvParams,
        backend: matmul.Backend,
    ) !Self {
        var engine = try matmul.MatmulEngine.init(allocator, backend, .f32);
        errdefer engine.deinit();

        const conv_kernel = params.convKernel();
        const scratch_conv = try allocator.alloc(f32, params.conv_dim * conv_kernel);
        errdefer allocator.free(scratch_conv);
        const scratch_in_proj = try allocator.alloc(f32, (3 * params.conv_dim) * params.n_embd);
        errdefer allocator.free(scratch_in_proj);
        const scratch_out_proj = try allocator.alloc(f32, params.n_embd * params.conv_dim);
        errdefer allocator.free(scratch_out_proj);

        const conv_state = try allocator.alloc(f32, params.convStateLen() * params.conv_dim);
        errdefer allocator.free(conv_state);
        @memset(conv_state, 0);

        var attn_norm = try Tensor(f32).alloc(allocator, &.{params.n_embd});
        errdefer attn_norm.deinit();
        var ffn_norm = try Tensor(f32).alloc(allocator, &.{params.n_embd});
        errdefer ffn_norm.deinit();

        return Self{
            .allocator = allocator,
            .layer_idx = layer_idx,
            .params = params,
            .matmul_engine = engine,
            .w_conv = undefined,
            .w_in_proj = undefined,
            .w_out_proj = undefined,
            .attn_norm = attn_norm,
            .ffn_norm = ffn_norm,
            .scratch_conv = scratch_conv,
            .scratch_in_proj = scratch_in_proj,
            .scratch_out_proj = scratch_out_proj,
            .conv_state = conv_state,
            .gpu = null,
        };
    }

    pub fn deinit(self: *Self) void {
        self.matmul_engine.deinit();
        self.allocator.free(self.scratch_conv);
        self.allocator.free(self.scratch_in_proj);
        self.allocator.free(self.scratch_out_proj);
        self.allocator.free(self.conv_state);
        self.attn_norm.deinit();
        self.ffn_norm.deinit();
        if (self.gpu) |*g| g.deinit();
    }

    pub fn resetState(self: *Self) void {
        @memset(self.conv_state, 0);
    }

    /// 8.3: análogos ShortConv de la interfaz snapshot/rollback SSM (lane-c
    /// C4.3) — el estado recurrente es g_conv_state (doble-buffer graph-safe
    /// @6541b3a: el vivo SIEMPRE queda en g_conv_state tras el copy-back).
    pub fn gpuStateLen(self: *const Self) usize {
        const g = self.gpu orelse return 0;
        return g.g_conv_state.len;
    }

    pub fn snapshotGpuState(self: *const Self, dst: []f32) !void {
        const g = self.gpu orelse return;
        if (dst.len < g.g_conv_state.len) return error.BufferTooSmall;
        try @import("cudaz").cuMemcpyDtoH(@intFromPtr(dst.ptr), @intFromPtr(g.g_conv_state.dev_ptr), g.g_conv_state.len * @sizeOf(f32));
    }

    pub fn restoreGpuState(self: *Self, src: []const f32) !void {
        const g = self.gpu orelse return;
        if (src.len < g.g_conv_state.len) return error.BufferTooSmall;
        try @import("cudaz").cuMemcpyHtoD(@intFromPtr(g.g_conv_state.dev_ptr), @intFromPtr(src.ptr), g.g_conv_state.len * @sizeOf(f32));
    }

    fn ensureScratch(self: *Self) !void {
        if (self.scratch_conv.len > 0) return;
        const p = self.params;
        const conv_kernel = p.convKernel();
        self.scratch_conv = try self.allocator.alloc(f32, p.conv_dim * conv_kernel);
        self.scratch_in_proj = try self.allocator.alloc(f32, (3 * p.conv_dim) * p.n_embd);
        self.scratch_out_proj = try self.allocator.alloc(f32, p.n_embd * p.conv_dim);
    }

    pub fn unloadWeights(self: *Self) void {
        // 7.1a-b (lane-c): evicción selectiva (ver hybrid_attn.unloadWeights)
        _ = self.matmul_engine.evictWeightCachePtr(@intFromPtr(self.scratch_conv.ptr));
        _ = self.matmul_engine.evictWeightCachePtr(@intFromPtr(self.scratch_in_proj.ptr));
        _ = self.matmul_engine.evictWeightCachePtr(@intFromPtr(self.scratch_out_proj.ptr));
        if (self.scratch_conv.len > 0) self.allocator.free(self.scratch_conv);
        if (self.scratch_in_proj.len > 0) self.allocator.free(self.scratch_in_proj);
        if (self.scratch_out_proj.len > 0) self.allocator.free(self.scratch_out_proj);
        self.scratch_conv = &[_]f32{};
        self.scratch_in_proj = &[_]f32{};
        self.scratch_out_proj = &[_]f32{};
    }

    pub fn loadWeightsFromGguf(self: *Self, g: *const gguf.GgufFile) !void {
        try self.ensureScratch();
        const prefix = try std.fmt.allocPrint(self.allocator, "blk.{d}.", .{self.layer_idx});
        defer self.allocator.free(prefix);

        // Depthwise conv1d weight: GGUF (kernel, conv_dim) — 8.3: la primera
        // dim ES el kernel size (ggml_ssm_conv: d_conv = c->ne[0]).
        self.w_conv = try loadQuantWeight(g, prefix, "shortconv.conv.weight");
        // 8.3: fijar el kernel real del tensor (LFM2.5: 3). ensureScratch()
        // se llamó con el fallback l_cache+1 (=4) — re-dimensionar al
        // tamaño real ANTES de dequantizar (si difiere).
        const kernel_real: usize = @intCast(self.w_conv.info.dims[0]);
        if (kernel_real != self.params.conv_kernel_gguf) {
            self.params.conv_kernel_gguf = kernel_real;
            if (self.scratch_conv.len != self.params.conv_dim * kernel_real) {
                if (self.scratch_conv.len > 0) self.allocator.free(self.scratch_conv);
                self.scratch_conv = try self.allocator.alloc(f32, self.params.conv_dim * kernel_real);
            }
            // 8.3 FIX: el conv_state host se alocó en init() con el fallback
            // (l_cache+1 ⇒ state_len = l_cache = 3 filas). Con el kernel real
            // el estado son kernel-1 filas — re-dimensionar para que el
            // upload a g_conv_state (mismo cálculo post-load) no reciba 6144
            // floats en un buffer device de 4096 (CudaMemcpyFailed).
            const want_state_len = self.params.convStateLen() * self.params.conv_dim;
            if (self.conv_state.len != want_state_len) {
                const new_state = try self.allocator.alloc(f32, want_state_len);
                @memset(new_state, 0);
                if (self.conv_state.len > 0) self.allocator.free(self.conv_state);
                self.conv_state = new_state;
            }
        }
        self.w_conv.dequantToF32Transposed(self.scratch_conv);

        // In projection (fused gate/up/value): [3*self.params.conv_dim, n_embd]
        self.w_in_proj = try loadQuantWeight(g, prefix, "shortconv.in_proj.weight");
        self.w_in_proj.dequantToF32Transposed(self.scratch_in_proj);

        // Out projection: [n_embd, self.params.conv_dim]
        self.w_out_proj = try loadQuantWeight(g, prefix, "shortconv.out_proj.weight");
        self.w_out_proj.dequantToF32Transposed(self.scratch_out_proj);

        // Norm weights
        self.attn_norm.deinit();
        self.attn_norm = try loadGgufF32(self.allocator, g, prefix, "attn_norm.weight");
        self.ffn_norm.deinit();
        self.ffn_norm = try loadGgufF32(self.allocator, g, prefix, "ffn_norm.weight");
    }

    /// Forward for prefill (N tokens) or single token decode
    pub fn forward(self: *Self, x: Tensor(f32), out: *Tensor(f32), n: usize) !void {
        const p = self.params;
        const conv_kernel = p.convKernel();
        const N = n;
        const sc_dbg = std.c.getenv("SC_DBG") != null;
        const scDumpHost = struct {
            fn go(tag: []const u8, data: []const f32) void {
                var s: f64 = 0;
                for (data) |v| s += @abs(@as(f64, v));
                debugz.dbg.print("[short_conv] cpu {s} sum|v|={d:.4} f0={d:.4} f1={d:.4} f2={d:.4}\n", .{ tag, s, data[0], data[1], data[2] });
            }
        };

        // === 1. Pre-norm (attn_norm) ===
        var norm_buf = try Tensor(f32).alloc(self.allocator, &.{ N, p.n_embd });
        defer norm_buf.deinit();
        @import("norm").rmsNorm(f32, f32, x, self.attn_norm, p.rms_eps, &norm_buf);
        if (sc_dbg) scDumpHost.go("norm", norm_buf.data);

        // === 2. In projection (fused b|c|x) — 8.3 LFM2 SEMÁNTICA REAL ======
        // Espejo del GPU path: in_proj ANTES de la conv; chunks (b|c|x) del
        // Gated DeltaNet (lfm2.cpp:174-196). La conv opera sobre b·x.
        // w_in_proj: [3*conv_dim, n_embd] → in_proj_out [N, 3*conv_dim]
        var in_proj_shape = [_]usize{ 3 * p.conv_dim, p.n_embd };
        var in_proj_strides = [_]usize{ p.n_embd, 1 };
        const w_in_proj32 = Tensor(f32){
            .data = self.scratch_in_proj,
            .shape = &in_proj_shape,
            .strides = &in_proj_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        var in_proj_out = try Tensor(f32).alloc(self.allocator, &.{ N, 3 * p.conv_dim });
        defer in_proj_out.deinit();
        try self.matmul_engine.linearProjection(f32, norm_buf, w_in_proj32, &in_proj_out);
        if (sc_dbg) scDumpHost.go("in_proj", in_proj_out.data);

        // === 3. Split [b | c | x] y gating GDN: bx = b · x ================
        var gated_out = try Tensor(f32).alloc(self.allocator, &.{ N, p.conv_dim });
        defer gated_out.deinit();
        for (0..N) |t| {
            for (0..p.conv_dim) |c| {
                const b = in_proj_out.data[t * (3 * p.conv_dim) + c];
                const xv = in_proj_out.data[t * (3 * p.conv_dim) + 2 * p.conv_dim + c];
                gated_out.data[t * p.conv_dim + c] = b * xv;
            }
        }

        // === 4. Depthwise conv1d causal SOBRE bx (lineal, sin silu) =======
        // LFM2 usa ggml_ssm_conv (lineal) — sin silu. Estado = kernel-1
        // filas (convStateLen): sx->ne[0] == d_conv-1+n_t (ggml.c:5568).
        const state_len = p.convStateLen();
        var conv_in = try Tensor(f32).alloc(self.allocator, &.{ state_len + N, p.conv_dim });
        defer conv_in.deinit();
        for (0..state_len) |t| {
            for (0..p.conv_dim) |c| {
                conv_in.data[t * p.conv_dim + c] = self.conv_state[t * p.conv_dim + c];
            }
        }
        for (0..N) |t| {
            for (0..p.conv_dim) |c| {
                conv_in.data[(state_len + t) * p.conv_dim + c] = gated_out.data[t * p.conv_dim + c];
            }
        }
        var conv_out = try Tensor(f32).alloc(self.allocator, &.{ N, p.conv_dim });
        defer conv_out.deinit();
        for (0..p.conv_dim) |c| {
            for (0..N) |t| {
                var sumf: f32 = 0;
                for (0..conv_kernel) |k| {
                    sumf += conv_in.data[(t + k) * p.conv_dim + c] * self.scratch_conv[c * conv_kernel + k];
                }
                // LFM2: conv LINEAL — sin silu (silu del GPU es bug).
                conv_out.data[t * p.conv_dim + c] = sumf;
            }
        }
        if (sc_dbg) {
            scDumpHost.go("bx", gated_out.data);
            scDumpHost.go("conv_out", conv_out.data);
        }
        // Update conv_state: últimas state_len filas de (state + input).
        for (0..state_len) |t| {
            for (0..p.conv_dim) |c| {
                self.conv_state[t * p.conv_dim + c] = conv_in.data[(N + t) * p.conv_dim + c];
            }
        }

        // === 5. Segundo gating: y = c · conv_out ===========================
        for (0..N) |t| {
            for (0..p.conv_dim) |c| {
                const c_gate = in_proj_out.data[t * (3 * p.conv_dim) + p.conv_dim + c];
                gated_out.data[t * p.conv_dim + c] = c_gate * conv_out.data[t * p.conv_dim + c];
            }
        }
        if (sc_dbg) scDumpHost.go("y_gated", gated_out.data);

        // === 5. Out projection ===
        // w_out_proj: [n_embd, self.params.conv_dim] -> output [N, n_embd]
        var out_proj_shape = [_]usize{ p.n_embd, p.conv_dim };
        var out_proj_strides = [_]usize{ p.conv_dim, 1 };
        const w_out_proj32 = Tensor(f32){
            .data = self.scratch_out_proj,
            .shape = &out_proj_shape,
            .strides = &out_proj_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        try self.matmul_engine.linearProjection(f32, gated_out, w_out_proj32, out);

        // === 6. Post-norm (ffn_norm) ===
        // Note: residual is added in HybridLayer.forward after this returns
        // The HybridLayer does: x + mixer_out, then post-norm
        // But for shortconv, we don't have a residual here - HybridLayer handles it
    }

    // ─── GPU Forward (Path B) ──────────────────────────────────────────────
    pub const ShortConvGpu = struct {
        g_norm: cublas.GpuTensor(f32),
        g_conv: cublas.GpuTensor(f32),
        g_in_proj: cublas.GpuTensor(f32),
        g_gated: cublas.GpuTensor(f32),
        g_out: cublas.GpuTensor(f32),
        g_conv_state: cublas.GpuBuffer(f32), // [convStateLen * conv_dim]
        // 8.3 FIX (race in-place): conv1dLinear leía conv_state mientras
        // otros hilos del MISMO launch escribían el nuevo estado sobre las
        // mismas direcciones (state_out == state). N=1: el hilo t=0 lee
        // state[i] que el hilo t=i ya sobrescribió ⇒ gating corrupto (paridad
        // scratch 70× off en decode). Estado doble-buffer + swap de puntero.
        g_conv_state_next: cublas.GpuBuffer(f32),
        g_conv_w: cublas.GpuBuffer(f32), // [self.params.conv_dim * conv_kernel]
        g_in_proj_w: cublas.GpuBuffer(f32), // [3*self.params.conv_dim * n_embd] (for device GEMM)
        g_out_proj_w: cublas.GpuBuffer(f32), // [n_embd * self.params.conv_dim] (for device GEMM)
        g_attn_norm: cublas.GpuBuffer(f32),
        g_ffn_norm: cublas.GpuBuffer(f32),
        cap_n: usize,
        params: ShortConvParams,

        fn alloc(p: ShortConvParams) !ShortConvGpu {
            const g_attn_norm = try cublas.GpuBuffer(f32).alloc(p.n_embd);
            const g_ffn_norm = try cublas.GpuBuffer(f32).alloc(p.n_embd);
            const g_conv_state = try cublas.GpuBuffer(f32).alloc(p.convStateLen() * p.conv_dim);
            const g_conv_state_next = try cublas.GpuBuffer(f32).alloc(p.convStateLen() * p.conv_dim);
            const g_conv_w = try cublas.GpuBuffer(f32).alloc(p.conv_dim * p.convKernel());
            const g_in_proj_w = try cublas.GpuBuffer(f32).alloc(3 * p.conv_dim * p.n_embd);
            const g_out_proj_w = try cublas.GpuBuffer(f32).alloc(p.n_embd * p.conv_dim);
            return .{
                .g_norm = try cublas.GpuTensor(f32).alloc(p.n_embd),
                .g_conv = try cublas.GpuTensor(f32).alloc(p.conv_dim),
                .g_in_proj = try cublas.GpuTensor(f32).alloc(3 * p.conv_dim),
                .g_gated = try cublas.GpuTensor(f32).alloc(p.conv_dim),
                .g_out = try cublas.GpuTensor(f32).alloc(p.n_embd),
                .g_conv_state = g_conv_state,
                .g_conv_state_next = g_conv_state_next,
                .g_conv_w = g_conv_w,
                .g_in_proj_w = g_in_proj_w,
                .g_out_proj_w = g_out_proj_w,
                .g_attn_norm = g_attn_norm,
                .g_ffn_norm = g_ffn_norm,
                .cap_n = 1,
                .params = p,
            };
        }

        fn ensureN(self: *ShortConvGpu, n: usize) !void {
            if (self.cap_n >= n) return;
            const p = self.params;
            if (self.cap_n > 0) {
                self.g_norm.deinit();
                self.g_conv.deinit();
                self.g_in_proj.deinit();
                self.g_gated.deinit();
                self.g_out.deinit();
            }
            self.g_norm = try cublas.GpuTensor(f32).alloc(n * p.n_embd);
            self.g_conv = try cublas.GpuTensor(f32).alloc(n * p.conv_dim);
            self.g_in_proj = try cublas.GpuTensor(f32).alloc(n * 3 * p.conv_dim);
            self.g_gated = try cublas.GpuTensor(f32).alloc(n * p.conv_dim);
            self.g_out = try cublas.GpuTensor(f32).alloc(n * p.n_embd);
            self.cap_n = n;
        }

        fn deinit(self: *ShortConvGpu) void {
            self.g_norm.deinit();
            self.g_conv.deinit();
            self.g_in_proj.deinit();
            self.g_gated.deinit();
            self.g_out.deinit();
            self.g_conv_state.free();
            self.g_conv_state_next.free();
            self.g_conv_w.free();
            self.g_in_proj_w.free();
            self.g_out_proj_w.free();
            self.g_attn_norm.free();
            self.g_ffn_norm.free();
        }
    };

    pub fn ensureGpu(self: *Self) !void {
        if (self.gpu != null) return;
        var g = try ShortConvGpu.alloc(self.params);
        const sc_dbg = std.c.getenv("SC_DBG") != null;
if (sc_dbg) debugz.dbg.print("[short_conv] ensureGpu li={d} uploads: attn_norm={d} ffn_norm={d} conv_state={d} scratch_conv={d} scratch_in={d} scratch_out={d}\n", .{ self.layer_idx, self.attn_norm.data.len, self.ffn_norm.data.len, self.conv_state.len, self.scratch_conv.len, self.scratch_in_proj.len, self.scratch_out_proj.len });
        try g.g_attn_norm.upload(self.attn_norm.data);
        try g.g_ffn_norm.upload(self.ffn_norm.data);
        try g.g_conv_state.upload(self.conv_state);
        try g.g_conv_w.upload(self.scratch_conv);
        // Upload in_proj weight (transposed for row-major GEMM)
        try self.uploadWeightToGpuTransposed(g.g_in_proj_w, self.scratch_in_proj);
        try self.uploadWeightToGpuTransposed(g.g_out_proj_w, self.scratch_out_proj);
        self.gpu = g;
    }

    fn uploadWeightToGpuTransposed(_: *Self, buf: cublas.GpuBuffer(f32), host_w: []f32) !void {
        // host_w is [out_dim, in_dim] in f32
        // We need to upload as-is for linearProjectionDevice (which expects [out, in] col-major)
        try buf.upload(host_w);
    }

    pub fn warmupGpuWeights(self: *Self) !void {
        // Warm up weight cache for in_proj and out_proj
        var in_proj_shape = [_]usize{ 3 * self.params.conv_dim, self.params.n_embd };
        var in_proj_strides = [_]usize{ self.params.n_embd, 1 };
        _ = try self.matmul_engine.projectionDevicePtr(Tensor(f32){ .data = self.scratch_in_proj, .shape = &in_proj_shape, .strides = &in_proj_strides, .offset = 0, .allocator = null, .owns_data = false });

        var out_proj_shape = [_]usize{ self.params.n_embd, self.params.conv_dim };
        var out_proj_strides = [_]usize{ self.params.conv_dim, 1 };
        _ = try self.matmul_engine.projectionDevicePtr(Tensor(f32){ .data = self.scratch_out_proj, .shape = &out_proj_shape, .strides = &out_proj_strides, .offset = 0, .allocator = null, .owns_data = false });
    }

    pub fn forwardGPU(
        self: *Self,
        lk: *layer_kernels.LayerKernels,
        x: cublas.GpuTensor(f32),
        out: *cublas.GpuTensor(f32),
        n: usize,
    ) !void {
        const p = self.params;
        const conv_kernel = p.convKernel();
        try ShortConvLayer.ensureGpu(self);
        const g = &self.gpu.?;
        try g.ensureN(n);

        // 8.3-diag (lane-f): traza por etapa — gate SC_DBG. Paridad CPU vs
        // GPU del shortconv (scratch). Coste cero sin flag.
        const sc_dbg = std.c.getenv("SC_DBG") != null;
        const scDump = struct {
            fn go(tag: []const u8, ptr: usize, elems: usize) void {
                const buf = std.heap.page_allocator.alloc(f32, elems) catch return;
                defer std.heap.page_allocator.free(buf);
                @import("cudaz").cuMemcpyDtoH(@intFromPtr(buf.ptr), ptr, elems * @sizeOf(f32)) catch return;
                var s: f64 = 0;
                for (buf) |v| s += @abs(@as(f64, v));
                debugz.dbg.print("[short_conv] gpu {s} sum|v|={d:.4} f0={d:.4} f1={d:.4} f2={d:.4}\n", .{ tag, s, buf[0], buf[1], buf[2] });
            }
        };

        // 1. Pre-norm (attn_norm)
        try lk.rmsNorm(x.ptr(), @intFromPtr(g.g_attn_norm.dev_ptr), g.g_norm.ptr(), n, p.n_embd, p.rms_eps);
        if (sc_dbg) scDump.go("norm", g.g_norm.ptr(), n * p.n_embd);

        // ── 8.3 LFM2 SEMÁNTICA REAL (lfm2.cpp:174-220) ──────────────────────
        // 2. In projection ANTES de la conv: split [b | c | x] del GDN.
        const qt_in = if (layer_kernels.quantPath() and !debugz.dbg.no_q4_ffn)
            qgemmTypeFor(self.w_in_proj.dtype())
        else
            null;
        if (qt_in) |qt| {
            try lk.qgemmLinear(self.allocator, g.g_norm.ptr(), self.w_in_proj.bytes, g.g_in_proj.ptr(), n, p.conv_dim, 3 * p.conv_dim, qt);
        } else {
            var in_proj_shape = [_]usize{ 3 * p.conv_dim, p.n_embd };
            var in_proj_strides = [_]usize{ p.n_embd, 1 };
            const w_in_proj32 = Tensor(f32){
                .data = self.scratch_in_proj,
                .shape = &in_proj_shape,
                .strides = &in_proj_strides,
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };
            try self.matmul_engine.linearProjectionDevice(g.g_norm, w_in_proj32, &g.g_in_proj, n, p.conv_dim, 3 * p.conv_dim);
        }
        if (sc_dbg) scDump.go("in_proj", g.g_in_proj.ptr(), n * 3 * p.conv_dim);

        // 3. Split [b|c|x] y primer gating: bx = b · x. 8.3 FIX: el in_proj
        // GEMM produce [T, 3*D] row-major INTERLEAVED por token (b/c/x son
        // slices de columna, no bloques globales [3][T][D]) — el mul con
        // punteros planos leía b=todo-el-bloque-0 (= b_0..b_{T-1} mal
        // alineado). gdnGate mode 0 indexa per-token.
        const chunk = n * p.conv_dim;
        const ip = g.g_in_proj.ptr();
        try lk.gdnGate(ip, 0, g.g_gated.ptr(), n, p.conv_dim, 0);
        if (sc_dbg) scDump.go("bx", g.g_gated.ptr(), chunk);

        // 4. Conv1d causal LINEAL sobre bx (LFM2 usa ggml_ssm_conv sin silu).
        // 8.3 FIX (race): state_out al buffer NEXT (no in-place) + copy-back
        // stream-ordered (capturable por CUDA graph: punteros A/B fijos, el
        // estado vivo siempre queda en g_conv_state tras la copia).
        try lk.conv1dLinear(
            @intFromPtr(g.g_conv_state.dev_ptr),
            g.g_gated.ptr(),
            @intFromPtr(g.g_conv_w.dev_ptr),
            g.g_conv.ptr(),
            @intFromPtr(g.g_conv_state_next.dev_ptr),
            n,
            p.conv_dim,
            conv_kernel,
        );
        try @import("cudaz").cuMemcpyDtoDAsync(
            @intFromPtr(g.g_conv_state.dev_ptr),
            @intFromPtr(g.g_conv_state_next.dev_ptr),
            p.convStateLen() * p.conv_dim * @sizeOf(f32),
            lk.stream,
        );

        // 5. Segundo gating: y = c · conv_out. 8.3 FIX: mismo layout
        // interleaved — gdnGate mode 1 (chunk 1 del in_proj · conv_out).
        try lk.gdnGate(ip, g.g_conv.ptr(), g.g_gated.ptr(), n, p.conv_dim, 1);
        if (sc_dbg) scDump.go("y_gated", g.g_gated.ptr(), chunk);

        // 6. Out projection on GPU
        // w_out_proj: [n_embd, conv_dim] from host scratch_out_proj
        // input: g_gated [n, conv_dim]
        // output: g_out [n, n_embd]
        // 8.3: mismo path cuantizado que el in_proj.
        // 6. Out projection on GPU
        // w_out_proj: [n_embd, conv_dim] from host scratch_out_proj
        // input: g_gated [n, conv_dim]
        // output: g_out [n, n_embd]
        // 8.3: mismo path cuantizado que el in_proj.
        const qt_out = if (layer_kernels.quantPath() and !debugz.dbg.no_q4_ffn)
            qgemmTypeFor(self.w_out_proj.dtype())
        else
            null;

        // 8.3 FIX (double ffn_norm): el forward YA no aplica ffn_norm al mixer
        // — lfm2.cpp:259-262 aplica ffn_norm UNA vez sobre el RESIDUAL
        // (x + mixer) en HybridLayer, igual que el CPU path (forward()
        // termina en out_proj raw). El rmsNorm final duplicaba la norma en
        // las 22 capas shortconv del LFM2.5 (las 8 attention ya iban bien):
        // ffn_norm(x + ffn_norm(mixer)) en vez de ffn_norm(x + mixer).
        // Las proyecciones escriben DIRECTO al buffer `out` del llamador.
        if (qt_out) |qt| {
            try lk.qgemmLinear(self.allocator, g.g_gated.ptr(), self.w_out_proj.bytes, out.ptr(), n, p.conv_dim, p.n_embd, qt);
        } else {
            var out_proj_shape = [_]usize{ p.n_embd, p.conv_dim };
            var out_proj_strides = [_]usize{ p.conv_dim, 1 };
            const w_out_proj32 = Tensor(f32){
                .data = self.scratch_out_proj,
                .shape = &out_proj_shape,
                .strides = &out_proj_strides,
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };
            try self.matmul_engine.linearProjectionDevice(g.g_gated, w_out_proj32, out, n, p.conv_dim, p.n_embd);
        }
    }

    fn gatedActivationGpu(self: *Self, lk: *layer_kernels.LayerKernels, in_proj: usize, out: usize, n: usize) !void {
        // in_proj: [n, 3*self.params.conv_dim] with [gate | up | value] each [n, self.params.conv_dim]
        // out: [n, self.params.conv_dim] = silu(gate) * up + value
        // We can use the swiglu kernel with a modified approach or write a custom kernel
        // For now, use the existing layer_kernels elementwise ops
        // The gateKernel does: out = sigmoid(x) * y
        // But we need: out = silu(gate) * up + value
        // We can use swiglu: silu(x) * y, then add value
        // Actually swiglu does: silu(x) * y
        // We have: gate, up, value
        // Want: silu(gate) * up + value
        // Can do: swiglu(gate, up) -> temp, then add value

        // For simplicity, use a custom approach: we can call swiglu on [gate, up] then add value
        // But we don't have an add kernel that works on the same buffer...
        // Let's use the existing kernels: gateKernel (sigmoid), then we need custom
        // Simplest: write a small kernel or use existing ops

        // Use the approach:
        // 1. gate = silu(gate) = gate / (1 + exp(-gate))
        // 2. temp = gate * up
        // 3. out = temp + value
        // We can use swiglu kernel for step 1+2 (it does silu(x)*y)
        // Then addKernel for step 3

        const gate_ptr = in_proj;
        const up_ptr = in_proj + n * self.params.conv_dim * @sizeOf(f32);
        const value_ptr = in_proj + 2 * n * self.params.conv_dim * @sizeOf(f32);

        // Step 1+2: swiglu(gate, up) -> g_gated
        try lk.swiglu(gate_ptr, up_ptr, n * self.params.conv_dim);
        // Step 3: add g_gated + value -> g_gated
        try lk.add(gate_ptr, value_ptr, out, n * self.params.conv_dim);
    }
};

fn loadQuantWeight(g: *const gguf.GgufFile, prefix: []const u8, name: []const u8) !QuantWeight {
    const full = try std.fmt.allocPrint(std.heap.page_allocator, "{s}{s}", .{ prefix, name });
    defer std.heap.page_allocator.free(full);
    const info = g.getTensor(full) orelse return ShortConvError.WeightFileNotFound;
    return QuantWeight.init(info, g.tensorData(info));
}

fn loadGgufF32(
    allocator: std.mem.Allocator,
    g: *const gguf.GgufFile,
    prefix: []const u8,
    name: []const u8,
) !Tensor(f32) {
    const full = try std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, name });
    defer allocator.free(full);
    const info = g.getTensor(full) orelse return ShortConvError.WeightFileNotFound;
    const numel: usize = @intCast(info.numel());

    const f32buf = try allocator.alloc(f32, numel);
    defer allocator.free(f32buf);
    try gguf.dequantTensor(info, g.tensorData(info), f32buf);

    var out_dim: usize = 1;
    var in_dim: usize = 1;
    var tensor: Tensor(f32) = undefined;
    if (info.n_dims >= 2) {
        in_dim = @intCast(info.dims[0]);
        out_dim = @intCast(info.dims[1]);
        tensor = try Tensor(f32).initUninitialized(allocator, &.{ out_dim, in_dim });
        for (0..in_dim) |r| {
            for (0..out_dim) |c| {
                tensor.data[c * in_dim + r] = f32buf[r + c * in_dim];
            }
        }
    } else {
        tensor = try Tensor(f32).initUninitialized(allocator, &.{numel});
        @memcpy(tensor.data, f32buf);
    }
    return tensor;
}
