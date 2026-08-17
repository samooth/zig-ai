//! Lanza los kernels elementwise de `layer_kernels.cu` vía la CUDA driver API.
//! Los kernels corren en el stream compartido para ordenarse con las GEMM de
//! cuBLAS (mismo stream); se sincroniza una sola vez por token desde el llamador.
const std = @import("std");
const cudaz = @import("cudaz");
const build_options = @import("build_options");

var g_module: ?cudaz.CUmodule = null;

/// Control global de la ruta de pesos cuantizados (Q4_0 GEMM device).
/// Se ajusta con `--quant`; los env `NOQ4*` siguen siendo un override.
pub var quant_enabled: bool = true;

pub fn quantPath() bool {
    return quant_enabled and std.c.getenv("NOQ4") == null;
}

fn loadModule() !cudaz.CUmodule {
    if (g_module) |m| return m;
    const cubin_path = build_options.layer_cubin;
    if (cubin_path.len == 0) return error.CudaUnavailable;
    try cudaz.ensureContext();
    g_module = try cudaz.cuModuleLoad(cubin_path);
    return g_module.?;
}

pub const LayerKernels = struct {
    stream: cudaz.CUstream,

    pub fn init(stream: cudaz.CUstream) !LayerKernels {
        _ = try loadModule();
        return .{ .stream = stream };
    }

    pub fn deinit(self: *LayerKernels) void {
        _ = self;
    }

    fn get(self: *LayerKernels, name: [:0]const u8) !cudaz.CUfunction {
        _ = self;
        return cudaz.cuModuleGetFunction(try loadModule(), name) catch return error.KernelNotFound;
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

    pub fn addInplace(self: *LayerKernels, a: usize, b: usize, n: usize) !void {
        var av = a;
        var bv = b;
        var n1: c_int = n_c(n);
        const func = try self.get("addInplaceKernel");
        var kp = [_]?*anyopaque{ &av, &bv, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn swiglu(self: *LayerKernels, gate: usize, up: usize, n: usize) !void {
        var gv = gate;
        var uv = up;
        var n1: c_int = n_c(n);
        const func = try self.get("swigluKernel");
        var kp = [_]?*anyopaque{ &gv, &uv, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn sigmoid(self: *LayerKernels, x: usize, n: usize) !void {
        var xv = x;
        var n1: c_int = n_c(n);
        const func = try self.get("sigmoidKernel");
        var kp = [_]?*anyopaque{ &xv, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn gateCompute(self: *LayerKernels, x: usize, dt_bias: usize, ssm_a: usize, n: usize, dt_rank: usize) !void {
        var xv = x;
        var dv = dt_bias;
        var sv = ssm_a;
        var n1: c_int = n_c(n);
        var dt1: c_int = n_c(dt_rank);
        const func = try self.get("gateComputeKernel");
        var kp = [_]?*anyopaque{ &xv, &dv, &sv, &n1, &dt1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
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

    pub fn conv1dSilu(self: *LayerKernels, conv_in: usize, conv_w: usize, conv_out: usize, N: usize, qkv_dim: usize, d_conv: usize) !void {
        var civ = conv_in;
        var cwv = conv_w;
        var cov = conv_out;
        var N1: c_int = n_c(N);
        var qv1: c_int = n_c(qkv_dim);
        var dc1: c_int = n_c(d_conv);
        const func = try self.get("conv1dSiluKernel");
        var kp = [_]?*anyopaque{ &civ, &cwv, &cov, &N1, &qv1, &dc1 };
        try cudaz.cuLaunchKernel(func, n_u((N * qkv_dim + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
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

    pub fn copyF32toF16(self: *LayerKernels, src: usize, dst: usize, n: usize) !void {
        var sv = src;
        var dv = dst;
        var n1: c_int = n_c(n);
        const func = try self.get("copyF32toF16Kernel");
        var kp = [_]?*anyopaque{ &sv, &dv, &n1 };
        try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
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

    pub fn mrope(self: *LayerKernels, data: usize, rows: usize, N: usize, head_dim: usize, n_rot: usize, base: f32, start_pos: usize) !void {
        var dv = data;
        var r1: c_int = n_c(rows);
        var N1: c_int = n_c(N);
        var hd1: c_int = n_c(head_dim);
        var nr1: c_int = n_c(n_rot);
        var bc: f32 = base;
        var sp: c_longlong = @intCast(start_pos);
        const func = try self.get("mropeKernel");
        var kp = [_]?*anyopaque{ &dv, &r1, &N1, &hd1, &nr1, &bc, &sp };
        try cudaz.cuLaunchKernel(func, n_u((rows + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    pub fn kvAppendF16(self: *LayerKernels, k: usize, v: usize, cache: usize, bt: usize, n: usize, start_pos: usize, kv_dim: usize, n_kv_head: usize, head_dim: usize, block_size: usize) !void {
        var kv = k;
        var vv = v;
        var cv = cache;
        var btv = bt;
        var n1: c_int = n_c(n);
        var sp: c_int = n_c(start_pos);
        var kvd1: c_int = n_c(kv_dim);
        var nkh1: c_int = n_c(n_kv_head);
        var hd1: c_int = n_c(head_dim);
        var bs1: c_int = n_c(block_size);
        const func = try self.get("kvAppendF16Kernel");
        var kp = [_]?*anyopaque{ &kv, &vv, &cv, &btv, &n1, &sp, &kvd1, &nkh1, &hd1, &bs1 };
        try cudaz.cuLaunchKernel(func, n_u((n * kv_dim + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
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

    /// Proyección Q4_0 M=1 con peso cuantizado: sube los bytes Q4_0 una sola vez
    /// (cache módulo) y lanza `q4gemmM1`. `w_bytes` = tensor GGUF [in,out] Q4_0.
    pub fn q4gemmLinear(self: *LayerKernels, allocator: std.mem.Allocator, x: usize, w_bytes: []const u8, out: usize, k: usize, n: usize) !void {
        const dev = try q4Weight(allocator, @intFromPtr(w_bytes.ptr), w_bytes);
        try self.q4gemmM1(x, dev, out, k, n);
    }
};

var q4_cache: ?std.AutoHashMap(usize, usize) = null;

fn q4Weight(allocator: std.mem.Allocator, key: usize, bytes: []const u8) !usize {
    if (q4_cache) |*m| {
        if (m.get(key)) |d| return d;
    }
    const dev = try cudaz.cuMemAlloc(bytes.len);
    try cudaz.cuMemcpyHtoD(dev, @intFromPtr(bytes.ptr), bytes.len);
    if (q4_cache == null) {
        q4_cache = std.AutoHashMap(usize, usize).init(allocator);
    }
    try q4_cache.?.put(key, dev);
    return dev;
}
