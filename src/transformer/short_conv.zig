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

pub const ShortConvError = error{
    WeightFileNotFound,
    ShapeMismatch,
};

pub const ShortConvParams = struct {
    n_embd: usize,
    conv_dim: usize,        // = n_embd (LFM2 uses same dim)
    l_cache: usize,         // kernel size - 1 (e.g., 3 for kernel=4)
    rms_eps: f32,

    pub fn convKernel(self: ShortConvParams) usize {
        return self.l_cache + 1;
    }
};

pub const ShortConvLayer = struct {
    allocator: std.mem.Allocator,
    layer_idx: usize,
    params: ShortConvParams,
    matmul_engine: matmul.MatmulEngine,

    // Weights
    w_conv: QuantWeight,      // [conv_dim, l_cache+1] depthwise conv1d
    w_in_proj: QuantWeight,   // [3*conv_dim, n_embd] fused gate/up/value
    w_out_proj: QuantWeight,  // [n_embd, conv_dim]

    // Norm weights (f32)
    attn_norm: Tensor(f32),   // pre-norm [n_embd]
    ffn_norm: Tensor(f32),    // post-conv norm [n_embd]

    // Scratch f32 (dequantized weights)
    scratch_conv: []f32,      // conv_dim * (l_cache+1)
    scratch_in_proj: []f32,   // (3*conv_dim) * n_embd
    scratch_out_proj: []f32,  // n_embd * conv_dim

    // Recurrent state for decode: conv_state [l_cache, conv_dim]
    conv_state: []f32,        // [l_cache * conv_dim]

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

        const conv_state = try allocator.alloc(f32, params.l_cache * params.conv_dim);
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

    fn ensureScratch(self: *Self) !void {
        if (self.scratch_conv.len > 0) return;
        const p = self.params;
        const conv_kernel = p.convKernel();
        self.scratch_conv = try self.allocator.alloc(f32, p.conv_dim * conv_kernel);
        self.scratch_in_proj = try self.allocator.alloc(f32, (3 * p.conv_dim) * p.n_embd);
        self.scratch_out_proj = try self.allocator.alloc(f32, p.n_embd * p.conv_dim);
    }

    pub fn unloadWeights(self: *Self) void {
        if (self.scratch_conv.len > 0) self.allocator.free(self.scratch_conv);
        if (self.scratch_in_proj.len > 0) self.allocator.free(self.scratch_in_proj);
        if (self.scratch_out_proj.len > 0) self.allocator.free(self.scratch_out_proj);
        self.scratch_conv = &[_]f32{};
        self.scratch_in_proj = &[_]f32{};
        self.scratch_out_proj = &[_]f32{};
        self.matmul_engine.clearWeightCache();
    }

    pub fn loadWeightsFromGguf(self: *Self, g: *const gguf.GgufFile) !void {
        try self.ensureScratch();
        const prefix = try std.fmt.allocPrint(self.allocator, "blk.{d}.", .{self.layer_idx});
        defer self.allocator.free(prefix);

        // Depthwise conv1d weight: [self.params.conv_dim, l_cache+1]
        self.w_conv = try loadQuantWeight(g, prefix, "shortconv.conv.weight");
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

        // === 1. Pre-norm (attn_norm) ===
        var norm_buf = try Tensor(f32).alloc(self.allocator, &.{ N, p.n_embd });
        defer norm_buf.deinit();
        @import("norm").rmsNorm(f32, f32, x, self.attn_norm, p.rms_eps, &norm_buf);

        // === 2. Depthwise conv1d + silu ===
        // Input: norm_buf [N, self.params.conv_dim] where self.params.conv_dim == n_embd
        // conv_state: [l_cache, self.params.conv_dim]
        // Weights: w_conv [self.params.conv_dim, conv_kernel] (channel-major: data[c*conv_kernel + k])
        var conv_in = try Tensor(f32).alloc(self.allocator, &.{ p.l_cache + N, p.conv_dim });
        defer conv_in.deinit();

        // Copy conv_state to beginning of conv_in
        for (0..p.l_cache) |t| {
            for (0..p.conv_dim) |c| {
                conv_in.data[t * p.conv_dim + c] = self.conv_state[t * p.conv_dim + c];
            }
        }
        // Copy current input
        for (0..N) |t| {
            for (0..p.conv_dim) |c| {
                conv_in.data[(p.l_cache + t) * p.conv_dim + c] = norm_buf.data[t * p.conv_dim + c];
            }
        }

        // conv1d + silu: output [N, self.params.conv_dim]
        var conv_out = try Tensor(f32).alloc(self.allocator, &.{ N, p.conv_dim });
        defer conv_out.deinit();

        // Weights are dequantized to scratch_conv [self.params.conv_dim, conv_kernel] (channel-major)
        for (0..p.conv_dim) |c| {
            for (0..N) |t| {
                var sumf: f32 = 0;
                for (0..conv_kernel) |k| {
                    sumf += conv_in.data[(t + k) * p.conv_dim + c] * self.scratch_conv[c * conv_kernel + k];
                }
                // silu activation
                conv_out.data[t * p.conv_dim + c] = sumf / (1.0 + @exp(-sumf));
            }
        }

        // Update conv_state with last l_cache rows of conv_in
        for (0..p.l_cache) |t| {
            for (0..p.conv_dim) |c| {
                self.conv_state[t * p.conv_dim + c] = conv_in.data[(N + t) * p.conv_dim + c];
            }
        }

        // === 3. In projection (fused gate/up/value) ===
        // w_in_proj: [3*self.params.conv_dim, n_embd] -> output [N, 3*self.params.conv_dim]
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
        try self.matmul_engine.linearProjection(f32, conv_out, w_in_proj32, &in_proj_out);

        // === 4. Split and gated activation ===
        // in_proj_out = [gate | up | value] each [N, self.params.conv_dim]
        // output = silu(gate) * up + value
        var gated_out = try Tensor(f32).alloc(self.allocator, &.{ N, p.conv_dim });
        defer gated_out.deinit();

        for (0..N) |t| {
            for (0..p.conv_dim) |c| {
                const gate = in_proj_out.data[t * (3 * p.conv_dim) + c];
                const up = in_proj_out.data[t * (3 * p.conv_dim) + p.conv_dim + c];
                const value = in_proj_out.data[t * (3 * p.conv_dim) + 2 * p.conv_dim + c];
                const silu_gate = gate / (1.0 + @exp(-gate));
                gated_out.data[t * p.conv_dim + c] = silu_gate * up + value;
            }
        }

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
        g_conv_state: cublas.GpuBuffer(f32),  // [l_cache * self.params.conv_dim]
        g_conv_w: cublas.GpuBuffer(f32),      // [self.params.conv_dim * conv_kernel]
        g_in_proj_w: cublas.GpuBuffer(f32),   // [3*self.params.conv_dim * n_embd] (for device GEMM)
        g_out_proj_w: cublas.GpuBuffer(f32),  // [n_embd * self.params.conv_dim] (for device GEMM)
        g_attn_norm: cublas.GpuBuffer(f32),
        g_ffn_norm: cublas.GpuBuffer(f32),
        cap_n: usize,
        params: ShortConvParams,

        fn alloc(p: ShortConvParams) !ShortConvGpu {
            const g_attn_norm = try cublas.GpuBuffer(f32).alloc(p.n_embd);
            const g_ffn_norm = try cublas.GpuBuffer(f32).alloc(p.n_embd);
            const g_conv_state = try cublas.GpuBuffer(f32).alloc(p.l_cache * p.conv_dim);
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
        // Weights already uploaded in ensureGpu
        _ = self;
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

        // 1. Pre-norm (attn_norm)
        try lk.rmsNorm(x.ptr(), @intFromPtr(g.g_attn_norm.dev_ptr), g.g_norm.ptr(), n, p.n_embd, p.rms_eps);

        // 2. Conv1d + silu on GPU
        // conv_state: [l_cache, self.params.conv_dim] (already on GPU in g_conv_state)
        // input: g_norm [n, self.params.conv_dim] where self.params.conv_dim == n_embd
        // conv_w: [self.params.conv_dim, conv_kernel] in g_conv_w
        // Output: g_conv [n, self.params.conv_dim]
        // New state: g_conv_state (updated in-place)
        try lk.conv1dSilu(
            @intFromPtr(g.g_conv_state.dev_ptr),
            g.g_norm.ptr(),
            @intFromPtr(g.g_conv_w.dev_ptr),
            g.g_conv.ptr(),
            @intFromPtr(g.g_conv_state.dev_ptr), // state_out reuses same buffer
            n, p.conv_dim, conv_kernel,
        );

        // 3. In projection (fused gate/up/value) on GPU
        // w_in_proj: [3*conv_dim, n_embd] from host scratch_in_proj
        // input: g_conv [n, conv_dim]
        // output: g_in_proj [n, 3*conv_dim]
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
        try self.matmul_engine.linearProjectionDevice(g.g_conv, w_in_proj32, &g.g_in_proj, n, p.conv_dim, 3 * p.conv_dim);

        // 4. Gated activation (silu(gate) * up + value) on GPU
        try self.gatedActivationGpu(lk, g.g_in_proj.ptr(), g.g_gated.ptr(), n);

        // 5. Out projection on GPU
        // w_out_proj: [n_embd, conv_dim] from host scratch_out_proj
        // input: g_gated [n, conv_dim]
        // output: g_out [n, n_embd]
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
        try self.matmul_engine.linearProjectionDevice(g.g_gated, w_out_proj32, &g.g_out, n, p.conv_dim, p.n_embd);

        // 6. Post-norm (ffn_norm) on GPU
        try lk.rmsNorm(g.g_out.ptr(), @intFromPtr(g.g_ffn_norm.dev_ptr), out.ptr(), n, p.n_embd, p.rms_eps);
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
        try lk.add(up_ptr, value_ptr, out, n * self.params.conv_dim);
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