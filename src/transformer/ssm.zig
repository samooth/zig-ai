//! SSM layer — Gated Delta Net (linear attention) para arquitecturas híbridas
//! tipo Qwen3.5 (qwen35). Implementación fiel a llama.cpp
//! (`qwen35.cpp::llm_build_qwen35::build_layer_attn_linear` +
//! `delta-net-base.cpp::build_delta_net_autoregressive` +
//! `ggml-cpu/ops.cpp::ggml_compute_forward_gated_delta_net_one_chunk`).
//!
//! Dims (Qwen3.5-9B): n_embd=4096, d_inner=4096, d_state=128, dt_rank=32,
//! n_group=16, conv_kernel=4.
//!
//!   key_dim   = d_state * n_group          = 2048   (Q, K)
//!   value_dim = d_state * dt_rank          = 4096   (V, z) = d_inner
//!   qkv_dim   = key_dim*2 + value_dim      = 8192   (attn_qkv out)
//!
//! Conv layout por token: [ q(2048) | k(2048) | v(4096) ], cabezas contiguas.
//! conv1d se almacena channel-major en el GGUF: data[c*d_conv + k].
//!
//! Recurrencia DeltaNet: para v-head hv con k-head hk = hv/(n_v/n_k) (bloque):
//!   g_h = softplus(alpha_proj + dt) * a     →  exp(g) escala el estado
//!   b_h = sigmoid(beta_proj)
//!   sk  = S^T k   ;  d = b_h * (v - sk)
//!   S  += k ⊗ d   ;  o = S^T q   (q escalado por 1/sqrt(d_state))
//! Luego rmsnorm(o, ssm_norm) * silu(z) y ssm_out.
//!
//! Estrategia de memoria: los pesos grandes (attn_qkv, attn_gate, ssm_out)
//! son QuantWeight sobre los bytes mmap; se dequantizan a scratch f16
//! persistente antes de cada matmul (dequant on-the-fly por matmul). Los
//! pesos pequeños (beta, alpha, dt, a, conv1d, ssm_norm) se cargan como
//! Tensor(f32) (son minúsculos y vienen f32/iq3_s pequeños).
const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const cublas = @import("cublas");
const cudaz = @import("cudaz");
const layer_kernels = @import("layer_kernels");
const gguf = @import("gguf");
const QuantWeight = @import("quant_weight").QuantWeight;
const debugz = @import("debug");

pub const SsmError = error{ WeightFileNotFound, ShapeMismatch };

pub const SsmParams = struct {
    n_embd: usize, // 4096
    d_inner: usize, // 4096 (value_dim)
    d_state: usize, // 128
    dt_rank: usize, // 32 (n_v_heads)
    n_group: usize, // 16 (n_k_heads)
    d_conv: usize, // 4
    rms_eps: f32 = 1e-6,

    pub fn keyDim(self: SsmParams) usize {
        return self.d_state * self.n_group;
    }

    pub fn qkvDim(self: SsmParams) usize {
        return self.d_state * self.n_group * 2 + self.d_inner;
    }
};

// PERF_SSM (lane-f F1): desglose por-etapa del decode SSM, acumulado en eventos
// CUDA con NOGRAPH+PERF_STAGE. Etapas: 0=qkv+z GEMV, 1=sigmoidGateProj,
// 2=conv1d+state-copy, 3=l2+deltaNet(+fused)+rmsGate, 4=out GEMV.
// Reporte: ssmStageReport() lo imprime main.zig junto a PERF (avg/token).
pub var ssm_stage_ns: [5]i128 = .{ 0, 0, 0, 0, 0 };
pub var ssm_stage_tokens: usize = 0;

pub fn ssmStageReport() void {
    if (ssm_stage_tokens == 0) return;
    const us: f64 = @floatFromInt(std.time.ns_per_us);
    const nt: f64 = @floatFromInt(ssm_stage_tokens);
    const names = [_][]const u8{ "qkv+z", "gate", "conv1d", "dn+rms", "out" };
    var total: i128 = 0;
    for (ssm_stage_ns) |v| total += v;
    debugz.dbg.print("[ssm] PERF_SSM avg/token/capa-sum: {d:.1} us\n", .{@as(f64, @floatFromInt(total)) / us / nt});
    for (names, ssm_stage_ns) |name, v| {
        debugz.dbg.print("[ssm]   {s: >8} {d: >8.1} us ({d:5.1}%)\n", .{ name, @as(f64, @floatFromInt(v)) / us / nt, @as(f64, @floatFromInt(v)) * 100.0 / @as(f64, @floatFromInt(total)) });
    }
}

pub const SsmLayer = struct {
    /// Buffers de activación persistentes para forward CPU (evitan alloc/free
    /// por llamada). Tamaño fijo para decode (N=1); prefill N>1 usa alloc
    /// dinámico porque los tensores crecen con N.
    const FwdBuffers = struct {
        qkv: []f32, // [qkv_dim]
        z: []f32, // [d_inner]
        beta: []f32, // [dt_rank]
        gate: []f32, // [dt_rank]
        conv_in: []f32, // [(d_conv-1+1) * qkv_dim] = d_conv * qkv_dim
        conv_out: []f32, // [qkv_dim]
        attn_out: []f32, // [d_inner]

        fn init(allocator: std.mem.Allocator, p: SsmParams) !FwdBuffers {
            const qkv_dim = p.qkvDim();
            return .{
                .qkv = try allocator.alloc(f32, qkv_dim),
                .z = try allocator.alloc(f32, p.d_inner),
                .beta = try allocator.alloc(f32, p.dt_rank),
                .gate = try allocator.alloc(f32, p.dt_rank),
                .conv_in = try allocator.alloc(f32, p.d_conv * qkv_dim),
                .conv_out = try allocator.alloc(f32, qkv_dim),
                .attn_out = try allocator.alloc(f32, p.d_inner),
            };
        }

        fn deinit(self: *FwdBuffers, allocator: std.mem.Allocator) void {
            allocator.free(self.qkv);
            allocator.free(self.z);
            allocator.free(self.beta);
            allocator.free(self.gate);
            allocator.free(self.conv_in);
            allocator.free(self.conv_out);
            allocator.free(self.attn_out);
        }
    };

    allocator: std.mem.Allocator,
    layer_idx: usize,
    params: SsmParams,
    matmul_engine: matmul.MatmulEngine,

    // Pesos grandes cuantizados → QuantWeight (bytes mmap, préstamo al GGUF).
    // Layout [out, in] para linearProjection (trans_b=true).
    w_qkv: QuantWeight, // [qkv_dim, n_embd]
    w_z: QuantWeight, // [value_dim, n_embd]
    w_out: QuantWeight, // [n_embd, value_dim]

    // Pesos pequeños (f32; los iq3_s [n_embd, dt_rank] son diminutos)
    w_beta: Tensor(f32), // [dt_rank, n_embd]
    w_alpha: Tensor(f32), // [dt_rank, n_embd]
    dt_bias: Tensor(f32), // [dt_rank]
    ssm_a: Tensor(f32), // [dt_rank]
    conv1d: Tensor(f32), // [qkv_dim, d_conv]  (channel-major en el GGUF)
    ssm_norm: Tensor(f32), // [d_state]

    // Scratch f16 persistente (reutilizado cada forward)
    scratch_qkv: []f32, // qkv_dim * n_embd
    scratch_z: []f32, // value_dim * n_embd
    scratch_out: []f32, // n_embd * value_dim

    // Buffers de activación persistentes para forward CPU (evitan alloc/free
    // por llamada). Se allocan una sola vez en init y se reutilizan.
    // Tamaño fijo para decode (N=1); prefill con N>1 cae al alloc dinámico.
    fwd_buf: ?FwdBuffers = null,

    // Estado recurrente
    conv_state: []f32, // [d_conv-1, qkv_dim] por secuencia
    s_state: []f32, // [d_state*d_state*dt_rank]  S[a][b][hv]

    // Estado y buffers GPU para el forward residente (Path B).
    gpu: ?GpuSsm = null,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        layer_idx: usize,
        params: SsmParams,
        backend: matmul.Backend,
    ) !Self {
        var engine = try matmul.MatmulEngine.init(allocator, backend, .f32);
        errdefer engine.deinit();

        // Scratch de pesos SSM f32 LAZY: se allocan en loadWeightsFromGguf.
        // Eager aquí costaba ~400MB/capa aunque la capa no estuviera cargada
        // (27B: OOM host durante la init de capas, antes del streaming).
        // conv_state/s_state sí son estado runtime y quedan eager (pequeños).

        const conv_state = try allocator.alloc(f32, (params.d_conv - 1) * params.qkvDim());
        errdefer allocator.free(conv_state);
        @memset(conv_state, 0);
        const s_state = try allocator.alloc(f32, params.d_state * params.d_state * params.dt_rank);
        errdefer allocator.free(s_state);
        @memset(s_state, 0);

        var w_beta = try Tensor(f32).alloc(allocator, &.{ params.dt_rank, params.n_embd });
        errdefer w_beta.deinit();
        var w_alpha = try Tensor(f32).alloc(allocator, &.{ params.dt_rank, params.n_embd });
        errdefer w_alpha.deinit();
        var dt_bias = try Tensor(f32).alloc(allocator, &.{params.dt_rank});
        errdefer dt_bias.deinit();
        var ssm_a = try Tensor(f32).alloc(allocator, &.{params.dt_rank});
        errdefer ssm_a.deinit();
        var conv1d = try Tensor(f32).alloc(allocator, &.{ params.qkvDim(), params.d_conv });
        errdefer conv1d.deinit();
        var ssm_norm = try Tensor(f32).alloc(allocator, &.{params.d_state});
        errdefer ssm_norm.deinit();

        const out_self = Self{
            .allocator = allocator,
            .layer_idx = layer_idx,
            .params = params,
            .matmul_engine = engine,
            .w_qkv = undefined,
            .w_z = undefined,
            .w_out = undefined,
            .w_beta = w_beta,
            .w_alpha = w_alpha,
            .dt_bias = dt_bias,
            .ssm_a = ssm_a,
            .conv1d = conv1d,
            .ssm_norm = ssm_norm,
            .scratch_qkv = &[_]f32{},
            .scratch_z = &[_]f32{},
            .scratch_out = &[_]f32{},
            .conv_state = conv_state,
            .s_state = s_state,
            .gpu = null,
            .fwd_buf = null,
        };
        return out_self;
    }

    pub fn deinit(self: *Self) void {
        if (self.gpu) |*g| g.deinit();
        self.matmul_engine.deinit();
        self.allocator.free(self.scratch_qkv);
        self.allocator.free(self.scratch_z);
        self.allocator.free(self.scratch_out);
        self.allocator.free(self.conv_state);
        self.allocator.free(self.s_state);
        if (self.fwd_buf) |*b| b.deinit(self.allocator);
        self.w_beta.deinit();
        self.w_alpha.deinit();
        self.dt_bias.deinit();
        self.ssm_a.deinit();
        self.conv1d.deinit();
        self.ssm_norm.deinit();
    }

    pub fn resetState(self: *Self) void {
        @memset(self.conv_state, 0);
        @memset(self.s_state, 0);
    }

    /// Re-alloca scratch si fue liberado por unloadWeights.
    fn ensureScratch(self: *Self) !void {
        if (self.scratch_qkv.len > 0) return;
        const p = self.params;
        const qkv_dim = p.qkvDim();
        self.scratch_qkv = try self.allocator.alloc(f32, qkv_dim * p.n_embd);
        self.scratch_z = try self.allocator.alloc(f32, p.d_inner * p.n_embd);
        self.scratch_out = try self.allocator.alloc(f32, p.n_embd * p.d_inner);
    }

    /// Inicializa los buffers persistentes de forward CPU (decode N=1).
    /// Para prefill N>1 se mantiene el alloc dinámico original.
    fn ensureFwdBuf(self: *Self) !void {
        if (self.fwd_buf != null) return;
        self.fwd_buf = try FwdBuffers.init(self.allocator, self.params);
    }

    /// Elementos f32 del estado recurrente GPU (s_state + conv_state).
    /// 0 si la capa no tiene estado GPU residente.
    pub fn gpuStateLen(self: *const Self) usize {
        const g = self.gpu orelse return 0;
        return g.d_s_state.len + g.d_conv_state.len;
    }

    /// Estado recurrente device → host (snapshot para rollback especulativo,
    /// lane-c C4.3). `dst` debe tener >= gpuStateLen() elementos.
    pub fn snapshotGpuState(self: *const Self, dst: []f32) !void {
        const g = self.gpu orelse return;
        if (dst.len < g.d_s_state.len + g.d_conv_state.len) return error.BufferTooSmall;
        try cudaz.cuMemcpyDtoH(@intFromPtr(dst.ptr), @intFromPtr(g.d_s_state.dev_ptr), g.d_s_state.len * @sizeOf(f32));
        try cudaz.cuMemcpyDtoH(@intFromPtr(dst.ptr + g.d_s_state.len), @intFromPtr(g.d_conv_state.dev_ptr), g.d_conv_state.len * @sizeOf(f32));
    }

    /// Restaura el estado recurrente desde un snapshot previo (host → device).
    pub fn restoreGpuState(self: *Self, src: []const f32) !void {
        const g = self.gpu orelse return;
        if (src.len < g.d_s_state.len + g.d_conv_state.len) return error.BufferTooSmall;
        try cudaz.cuMemcpyHtoD(@intFromPtr(g.d_s_state.dev_ptr), @intFromPtr(src.ptr), g.d_s_state.len * @sizeOf(f32));
        try cudaz.cuMemcpyHtoD(@intFromPtr(g.d_conv_state.dev_ptr), @intFromPtr(src.ptr + g.d_s_state.len), g.d_s_state.len * @sizeOf(f32));
    }

    /// P0-1 (dev RLT): zero del estado recurrente RESIDENTE (d_s_state +
    /// d_conv_state) sin pasar por host. Para PPL sliding-window: cada chunk
    /// reconstruye el estado recurrente desde ctx_start (semántica llama.cpp
    /// — resetState() host NO toca los buffers device).
    pub fn zeroGpuState(self: *Self) void {
        const g = self.gpu orelse return;
        cudaz.cuMemsetD8(@intFromPtr(g.d_s_state.dev_ptr), 0, g.d_s_state.len * @sizeOf(f32)) catch {};
        cudaz.cuMemsetD8(@intFromPtr(g.d_conv_state.dev_ptr), 0, g.d_conv_state.len * @sizeOf(f32)) catch {};
    }

    /// Libera scratch f32 dequantizados. Requiere dequantización en loadWeightsFromGguf.
    pub fn unloadWeights(self: *Self) void {
        // 7.1a-b (lane-c): evicción selectiva (ver hybrid_attn.unloadWeights)
        _ = self.matmul_engine.evictWeightCachePtr(@intFromPtr(self.scratch_qkv.ptr));
        _ = self.matmul_engine.evictWeightCachePtr(@intFromPtr(self.scratch_z.ptr));
        _ = self.matmul_engine.evictWeightCachePtr(@intFromPtr(self.scratch_out.ptr));
        if (self.scratch_qkv.len > 0) self.allocator.free(self.scratch_qkv);
        if (self.scratch_z.len > 0) self.allocator.free(self.scratch_z);
        if (self.scratch_out.len > 0) self.allocator.free(self.scratch_out);
        self.scratch_qkv = &[_]f32{};
        self.scratch_z = &[_]f32{};
        self.scratch_out = &[_]f32{};
    }

    /// Carga los pesos desde el GGUF (nombres qwen35). Los grandes quedan
    /// como QuantWeight (bytes mmap, sin copia); los pequeños se dequantizan
    /// a f32.
    /// T1 fix regresión CPU (bisect D 02:5x): los consumidores f32 del
    /// scratch (forward CPU y fallback GPU) garantizan alloc+dequant bajo
    /// demanda — el gating de loadWeightsFromGguf puede haberlos saltado.
    fn ensureScratchFilled(self: *Self) !void {
        try self.ensureScratch();
        self.w_qkv.dequantToF32Transposed(self.scratch_qkv);
        self.w_z.dequantToF32Transposed(self.scratch_z);
        self.w_out.dequantToF32Transposed(self.scratch_out);
    }

    pub fn loadWeightsFromGguf(self: *SsmLayer, g: *const gguf.GgufFile) !void {
        if (debugz.dbg.dump_prefill_layers) {
            debugz.load_count += 1;
            debugz.dbg.print("[ssm] SSM_LOAD li={d} call={d} wbeta_old={*}\n", .{ self.layer_idx, debugz.load_count, self.w_beta.data.ptr });
        }
        const prefix = try std.fmt.allocPrint(self.allocator, "blk.{d}.", .{self.layer_idx});
        defer self.allocator.free(prefix);

        self.w_qkv = try loadQuantWeight(g, prefix, "attn_qkv.weight");
        self.w_z = try loadQuantWeight(g, prefix, "attn_gate.weight");
        self.w_out = try loadQuantWeight(g, prefix, "ssm_out.weight");

        // T1 VRAM-spec: dequantizar los grandes UNA vez SOLO si el camino
        // cuantizado no cubre los tres pesos (con kernel qgemm los scratch f32
        // no se leen nunca y con streaming acumulaban ~460MB × capas hasta el
        // OOM del host — el streamer solo expulsa por presión de VRAM).
        const qt_qkv_l = qgemmTypeFor(self.w_qkv.dtype());
        const qt_z_l = qgemmTypeFor(self.w_z.dtype());
        const qt_out_l = qgemmTypeFor(self.w_out.dtype());
        const need_f32 = !(quantSsmEnabled() and qt_qkv_l != null and qt_z_l != null and qt_out_l != null);
        if (need_f32) {
            try self.ensureScratch();
            // Dequantizar los pesos grandes UNA vez aquí (no por token en forward),
            // igual que hace la FFN. Los scratch f32 persisten y el caché de pesos
            // GPU los sube al device una sola vez.
            self.w_qkv.dequantToF32Transposed(self.scratch_qkv);
            self.w_z.dequantToF32Transposed(self.scratch_z);
            self.w_out.dequantToF32Transposed(self.scratch_out);
        }
        self.w_beta.deinit();
        self.w_beta = try loadGgufF32(self.allocator, g, prefix, "ssm_beta.weight", true);
        self.w_alpha.deinit();
        self.w_alpha = try loadGgufF32(self.allocator, g, prefix, "ssm_alpha.weight", true);
        self.dt_bias.deinit();
        self.dt_bias = try loadGgufF32(self.allocator, g, prefix, "ssm_dt.bias", false);
        self.ssm_a.deinit();
        self.ssm_a = try loadGgufF32(self.allocator, g, prefix, "ssm_a", false);
        self.conv1d.deinit();
        // conv1d GGUF es [d_conv, conv_dim]; se indexa channel-major
        // conv1d.data[c*d_conv+k] → necesita el layout transpuesto.
        self.conv1d = try loadGgufF32(self.allocator, g, prefix, "ssm_conv1d.weight", true);
        self.ssm_norm.deinit();
        self.ssm_norm = try loadGgufF32(self.allocator, g, prefix, "ssm_norm.weight", false);
    }

    /// Forward del bloque SSM (gated delta net).
    /// `x`: Tensor(f16) [1, N, n_embd] position-major.
    /// `out`: Tensor(f16) [1, N, n_embd].
    /// `n`: tokens a procesar (prefill en bloque o 1 token de generación).
    pub fn forward(self: *SsmLayer, x: Tensor(f32), out: *Tensor(f32), n: usize) !void {
        const p = self.params;
        const qkv_dim = p.qkvDim();
        const key_dim = p.keyDim();
        const n_k_heads = p.n_group;
        const n_v_heads = p.dt_rank;
        const head_v_dim = p.d_state;
        const N = n;

        // Buffers persistentes para decode (N=1): eliminan alloc/free por token.
        // Prefill N>1 mantiene el alloc dinámico original.
        const use_persistent = N == 1;
        if (use_persistent) try self.ensureFwdBuf();
        const fb = if (use_persistent) self.fwd_buf.? else null;

        // 1. qkv = attn_qkv @ X → [N, qkv_dim]
        // T1 fix regresión CPU: consumidor f32 garantiza scratch lleno.
        try self.ensureScratchFilled();
        var w_qkv_shape = [_]usize{ qkv_dim, p.n_embd };
        var w_qkv_strides = [_]usize{ p.n_embd, 1 };
        const w_qkv32 = Tensor(f32){
            .data = self.scratch_qkv,
            .shape = &w_qkv_shape,
            .strides = &w_qkv_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        const qkv_data = if (use_persistent) fb.?.qkv else try self.allocator.alloc(f32, N * qkv_dim);
        defer if (!use_persistent) self.allocator.free(qkv_data);
        var qkv_shape = [_]usize{ N, qkv_dim };
        var qkv_strides = [_]usize{ qkv_dim, 1 };
        var qkv = Tensor(f32){ .data = qkv_data, .shape = &qkv_shape, .strides = &qkv_strides, .offset = 0, .allocator = null, .owns_data = false };
        try self.matmul_engine.linearProjection(f32, x, w_qkv32, &qkv);

        // 2. z = attn_gate @ X → [N, value_dim] (peso ya dequantizado en load)
        var w_z_shape = [_]usize{ p.d_inner, p.n_embd };
        var w_z_strides = [_]usize{ p.n_embd, 1 };
        const w_z32 = Tensor(f32){
            .data = self.scratch_z,
            .shape = &w_z_shape,
            .strides = &w_z_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        const z_data = if (use_persistent) fb.?.z else try self.allocator.alloc(f32, N * p.d_inner);
        defer if (!use_persistent) self.allocator.free(z_data);
        var z_shape = [_]usize{ N, p.d_inner };
        var z_strides = [_]usize{ p.d_inner, 1 };
        var z = Tensor(f32){ .data = z_data, .shape = &z_shape, .strides = &z_strides, .offset = 0, .allocator = null, .owns_data = false };
        try self.matmul_engine.linearProjection(f32, x, w_z32, &z);

        // 3. beta = sigmoid(ssm_beta @ X) → [N, dt_rank]
        const beta_data = if (use_persistent) fb.?.beta else try self.allocator.alloc(f32, N * p.dt_rank);
        defer if (!use_persistent) self.allocator.free(beta_data);
        var beta_shape = [_]usize{ N, p.dt_rank };
        var beta_strides = [_]usize{ p.dt_rank, 1 };
        var beta = Tensor(f32){ .data = beta_data, .shape = &beta_shape, .strides = &beta_strides, .offset = 0, .allocator = null, .owns_data = false };
        try self.matmul_engine.linearProjection(f32, x, self.w_beta, &beta);
        for (beta.data) |*v| v.* = 1.0 / (1.0 + @exp(-v.*));

        // 4. gate = softplus(ssm_alpha @ X + dt) * ssm_a → [N, dt_rank]
        //    El GGUF guarda ssm_a YA como -exp(A_log); el decay es exp(gate).
        const gate_data = if (use_persistent) fb.?.gate else try self.allocator.alloc(f32, N * p.dt_rank);
        defer if (!use_persistent) self.allocator.free(gate_data);
        var gate_shape = [_]usize{ N, p.dt_rank };
        var gate_strides = [_]usize{ p.dt_rank, 1 };
        var gate = Tensor(f32){ .data = gate_data, .shape = &gate_shape, .strides = &gate_strides, .offset = 0, .allocator = null, .owns_data = false };
        try self.matmul_engine.linearProjection(f32, x, self.w_alpha, &gate);
        for (0..p.dt_rank) |h| {
            for (0..N) |t| {
                const v = gate.data[t * p.dt_rank + h] + self.dt_bias.data[h];
                const sp = @log(1.0 + @exp(v));
                gate.data[t * p.dt_rank + h] = self.ssm_a.data[h] * sp;
            }
        }

        // 5. Conv causal + silu → conv_out [N, qkv_dim]
        const conv_in_data = if (use_persistent) fb.?.conv_in else try self.allocator.alloc(f32, (p.d_conv - 1 + N) * qkv_dim);
        defer if (!use_persistent) self.allocator.free(conv_in_data);
        var conv_in_shape = [_]usize{ (p.d_conv - 1) + N, qkv_dim };
        var conv_in_strides = [_]usize{ qkv_dim, 1 };
        var conv_in = Tensor(f32){ .data = conv_in_data, .shape = &conv_in_shape, .strides = &conv_in_strides, .offset = 0, .allocator = null, .owns_data = false };
        for (0..p.d_conv - 1) |t| {
            for (0..qkv_dim) |c| conv_in.data[t * qkv_dim + c] = self.conv_state[t * qkv_dim + c];
        }
        for (0..N) |t| {
            for (0..qkv_dim) |c| conv_in.data[(p.d_conv - 1 + t) * qkv_dim + c] = qkv.data[t * qkv_dim + c];
        }
        const conv_out_data = if (use_persistent) fb.?.conv_out else try self.allocator.alloc(f32, N * qkv_dim);
        defer if (!use_persistent) self.allocator.free(conv_out_data);
        var conv_out_shape = [_]usize{ N, qkv_dim };
        var conv_out_strides = [_]usize{ qkv_dim, 1 };
        var conv_out = Tensor(f32){ .data = conv_out_data, .shape = &conv_out_shape, .strides = &conv_out_strides, .offset = 0, .allocator = null, .owns_data = false };
        for (0..qkv_dim) |c| {
            for (0..N) |t| {
                var sumf: f32 = 0;
                for (0..p.d_conv) |k| {
                    // conv1d channel-major: data[c*d_conv + k]
                    sumf += conv_in.data[(t + k) * qkv_dim + c] * self.conv1d.data[c * p.d_conv + k];
                }
                conv_out.data[t * qkv_dim + c] = sumf / (1.0 + @exp(-sumf));
            }
        }
        // actualizar conv_state con las últimas d_conv-1 filas
        for (0..p.d_conv - 1) |t| {
            for (0..qkv_dim) |c| {
                self.conv_state[t * qkv_dim + c] = conv_in.data[(N + t) * qkv_dim + c];
            }
        }

        // 5b. L2-norm de Q y K por head (scale = 1/sqrt(sum x^2 + eps) — FLA
        // l2norm.py:39/69; 1.14: eps DENTRO de la raíz, NO clamp)
        self.l2NormQK(conv_out, N, key_dim, n_k_heads, head_v_dim);

        // 6. Recurrencia DeltaNet token por token → attn_out [N, value_dim]
        const attn_out_data = if (use_persistent) fb.?.attn_out else try self.allocator.alloc(f32, N * p.d_inner);
        defer if (!use_persistent) self.allocator.free(attn_out_data);
        var attn_out_shape = [_]usize{ N, p.d_inner };
        var attn_out_strides = [_]usize{ p.d_inner, 1 };
        var attn_out = Tensor(f32){ .data = attn_out_data, .shape = &attn_out_shape, .strides = &attn_out_strides, .offset = 0, .allocator = null, .owns_data = false };
        self.deltaNetRecurrence(conv_out, gate, beta, attn_out, N, key_dim, n_k_heads, n_v_heads, head_v_dim);

        // 7. rmsnorm(attn_out, ssm_norm) * silu(z) (por v-head, dims 128 contiguas)
        for (0..n_v_heads) |hv| {
            for (0..N) |t| {
                const base = t * p.d_inner + hv * head_v_dim;
                var mean_sq: f32 = 0;
                for (0..head_v_dim) |i| {
                    const v = attn_out.data[base + i];
                    mean_sq += v * v;
                }
                mean_sq /= @as(f32, @floatFromInt(head_v_dim));
                const rscale = 1.0 / @sqrt(mean_sq + p.rms_eps);
                for (0..head_v_dim) |i| {
                    const zn = z.data[base + i];
                    const silu = zn / (1.0 + @exp(-zn));
                    attn_out.data[base + i] = attn_out.data[base + i] * rscale * self.ssm_norm.data[i] * silu;
                }
            }
        }

        // 8. out = ssm_out @ attn_out → [N, n_embd]
        // T1: el peso puede NO estar dequantizado si el gating lo saltó
        // (camino GPU cuantizado) — el consumidor f32 lo garantiza aquí.
        try self.ensureScratchFilled();
        var w_out_shape = [_]usize{ p.n_embd, p.d_inner };
        var w_out_strides = [_]usize{ p.d_inner, 1 };
        const w_out32 = Tensor(f32){
            .data = self.scratch_out,
            .shape = &w_out_shape,
            .strides = &w_out_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        try self.matmul_engine.linearProjection(f32, attn_out, w_out32, out);
    }

    /// L2-normaliza cada head de Q y K en conv_out (fiel a ggml_l2_norm).
    fn l2NormQK(
        self: *SsmLayer,
        conv_out: Tensor(f32),
        N: usize,
        key_dim: usize,
        n_k_heads: usize,
        head_v_dim: usize,
    ) void {
        const qkv_dim = self.params.qkvDim();
        for (0..N) |t| {
            for (0..n_k_heads) |h| {
                const q_base = t * qkv_dim + h * head_v_dim;
                const k_base = t * qkv_dim + key_dim + h * head_v_dim;
                for (0..2) |part| {
                    const base = if (part == 0) q_base else k_base;
                    var sum_sq: f32 = 0;
                    for (0..head_v_dim) |i| {
                        const v = conv_out.data[base + i];
                        sum_sq += v * v;
                    }
                    // 1.14: fórmula FLA l2norm.py:69 `x/sqrt(sum+eps)` — eps
                    // dentro de la raíz. El clamp anterior (1/max(sqrt,eps))
                    // divergía de FLA cuando sum < eps² (q/k casi-nulos tras
                    // silu con beta baja en decode largo).
                    const scale = 1.0 / @sqrt(sum_sq + self.params.rms_eps);
                    for (0..head_v_dim) |i| conv_out.data[base + i] *= scale;
                }
            }
        }
    }

    /// Recurrencia DeltaNet (fiel al kernel `gated_delta_net_one_chunk`).
    /// `conv_out`: [N, qkv_dim] → q en [0,key_dim), k en [key_dim,2*key_dim), v en [2*key_dim,qkv_dim).
    fn deltaNetRecurrence(
        self: *SsmLayer,
        conv_out: Tensor(f32),
        gate: Tensor(f32),
        beta: Tensor(f32),
        attn_out: Tensor(f32),
        N: usize,
        key_dim: usize,
        n_k_heads: usize,
        n_v_heads: usize,
        head_v_dim: usize,
    ) void {
        const p = self.params;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_v_dim)));

        for (0..N) |t| {
            for (0..n_v_heads) |hv| {
                // bloque: la k-head de la v-head hv es hv/(n_v/n_k) (repeat_interleave)
                const hk = hv % n_k_heads; // 7.1b: módulo (ggml_repeat_4d del oráculo), no bloque
                const g = @exp(gate.data[t * p.dt_rank + hv]);
                const b = beta.data[t * p.dt_rank + hv];

                const q_base = t * p.qkvDim() + hk * head_v_dim;
                const k_base = t * p.qkvDim() + key_dim + hk * head_v_dim;
                const v_base = t * p.qkvDim() + 2 * key_dim + hv * head_v_dim;
                const s_base = hv * head_v_dim * head_v_dim;

                // S *= exp(g)
                for (0..head_v_dim * head_v_dim) |i| self.s_state[s_base + i] *= g;

                // sk[j] = sum_i S[i][j] * k[i] ; S[a][b] en s_state[a*S_v + b]
                var d_buf: [128]f32 = undefined;
                for (0..head_v_dim) |j| {
                    var sk: f32 = 0;
                    for (0..head_v_dim) |i| {
                        sk += self.s_state[s_base + i * head_v_dim + j] * conv_out.data[k_base + i];
                    }
                    d_buf[j] = b * (conv_out.data[v_base + j] - sk);
                }
                // S[i][j] += k[i] * d[j]
                for (0..head_v_dim) |i| {
                    const kv = conv_out.data[k_base + i];
                    for (0..head_v_dim) |j| {
                        self.s_state[s_base + i * head_v_dim + j] += kv * d_buf[j];
                    }
                }
                // o[j] = sum_i S[i][j] * q[i] * scale
                for (0..head_v_dim) |j| {
                    var o: f32 = 0;
                    for (0..head_v_dim) |i| {
                        o += self.s_state[s_base + i * head_v_dim + j] * conv_out.data[q_base + i];
                    }
                    attn_out.data[t * p.d_inner + hv * head_v_dim + j] = o * scale;
                }
            }
        }
    }

    // ─── Forward SSM residente en GPU (Path B) ─────────────────────────────────
    // Toda la activación vive en device: matmuls device→device (pesos cacheados en
    // GPU) + kernels elementwise. El estado recurrente (s_state, conv_state) también
    // vive en GPU y persiste entre tokens. El llamador sincroniza el stream.
    pub const GpuSsm = struct {
        d_dt_bias: cublas.GpuBuffer(f32),
        d_ssm_a: cublas.GpuBuffer(f32),
        d_conv1d: cublas.GpuBuffer(f32),
        d_ssm_norm: cublas.GpuBuffer(f32),
        d_s_state: cublas.GpuBuffer(f32),
        d_conv_state: cublas.GpuBuffer(f32),
        qkv: cublas.GpuTensor(f32),
        z: cublas.GpuTensor(f32),
        beta: cublas.GpuTensor(f32),
        gate: cublas.GpuTensor(f32),
        conv_in: cublas.GpuTensor(f32),
        conv_out: cublas.GpuTensor(f32),
        attn_out: cublas.GpuTensor(f32),
        cap_n: usize,
        params: SsmParams,

        fn alloc(p: SsmParams, dt_bias: []f32, ssm_a: []f32, conv1d: []f32, ssm_norm: []f32) !GpuSsm {
            const d_dt_bias = try cublas.GpuBuffer(f32).alloc(dt_bias.len);
            try d_dt_bias.upload(dt_bias);
            const d_ssm_a = try cublas.GpuBuffer(f32).alloc(ssm_a.len);
            try d_ssm_a.upload(ssm_a);
            const d_conv1d = try cublas.GpuBuffer(f32).alloc(conv1d.len);
            try d_conv1d.upload(conv1d);
            const d_ssm_norm = try cublas.GpuBuffer(f32).alloc(ssm_norm.len);
            try d_ssm_norm.upload(ssm_norm);
            const d_s_state = try cublas.GpuBuffer(f32).alloc(p.d_state * p.d_state * p.dt_rank);
            try cudaz.cuMemsetD8(@intFromPtr(d_s_state.dev_ptr), 0, p.d_state * p.d_state * p.dt_rank * @sizeOf(f32));
            const d_conv_state = try cublas.GpuBuffer(f32).alloc((p.d_conv - 1) * p.qkvDim());
            try cudaz.cuMemsetD8(@intFromPtr(d_conv_state.dev_ptr), 0, (p.d_conv - 1) * p.qkvDim() * @sizeOf(f32));
            return .{
                .d_dt_bias = d_dt_bias,
                .d_ssm_a = d_ssm_a,
                .d_conv1d = d_conv1d,
                .d_ssm_norm = d_ssm_norm,
                .d_s_state = d_s_state,
                .d_conv_state = d_conv_state,
                .qkv = undefined,
                .z = undefined,
                .beta = undefined,
                .gate = undefined,
                .conv_in = undefined,
                .conv_out = undefined,
                .attn_out = undefined,
                .cap_n = 0,
                .params = p,
            };
        }

        fn ensureN(self: *GpuSsm, n: usize) !void {
            if (self.cap_n >= n) return;
            const p = self.params;
            const qkv_dim = p.qkvDim();
            if (self.cap_n > 0) {
                self.qkv.deinit();
                self.z.deinit();
                self.beta.deinit();
                self.gate.deinit();
                self.conv_in.deinit();
                self.conv_out.deinit();
                self.attn_out.deinit();
            }
            self.qkv = try cublas.GpuTensor(f32).alloc(n * qkv_dim);
            self.z = try cublas.GpuTensor(f32).alloc(n * p.d_inner);
            self.beta = try cublas.GpuTensor(f32).alloc(n * p.dt_rank);
            self.gate = try cublas.GpuTensor(f32).alloc(n * p.dt_rank);
            self.conv_in = try cublas.GpuTensor(f32).alloc((p.d_conv - 1 + n) * qkv_dim);
            self.conv_out = try cublas.GpuTensor(f32).alloc(n * qkv_dim);
            self.attn_out = try cublas.GpuTensor(f32).alloc(n * p.d_inner);
            self.cap_n = n;
        }

        fn deinit(self: *GpuSsm) void {
            self.d_dt_bias.free();
            self.d_ssm_a.free();
            self.d_conv1d.free();
            self.d_ssm_norm.free();
            self.d_s_state.free();
            self.d_conv_state.free();
            if (self.cap_n > 0) {
                self.qkv.deinit();
                self.z.deinit();
                self.beta.deinit();
                self.gate.deinit();
                self.conv_in.deinit();
                self.conv_out.deinit();
                self.attn_out.deinit();
            }
        }
    };

    pub fn ensureGpu(self: *SsmLayer) !void {
        if (self.gpu != null) return;
        self.gpu = try GpuSsm.alloc(self.params, self.dt_bias.data, self.ssm_a.data, self.conv1d.data, self.ssm_norm.data);
    }

    /// T1 VRAM-spec (ticket lane-c 21:20): tipo qgemmKernel para el peso, o
    /// null si no hay kernel cuantizado (⇒ fallback f32). Antes solo q4_0/q5_k
    /// usaban el camino cuantizado y el resto dequantizaba a W_T f32
    /// (d_inner×n_embd ≈126-250MB por capa y proyección) en weight_cache SIN
    /// eviction ⇒ pico post-prefill que mataba el bootstrap 27B con streaming.
    pub fn qgemmTypeFor(t: gguf.GgmlType) ?u32 {
        return switch (t) {
            .q4_0 => 0,
            .q4_1 => 1,
            .q5_k => 2,
            .q6_k => 3,
            .q4_k => 4,
            .q8_0 => 5,
            .q3_k => 6,
            .q2_k => 7,
            // T1-B (lane-b @ac05be3): type 8 = iq3_s — desbloquea el 9B
            // IQ3_M (sus tensores iq3_s dejaban de caer al fallback f32).
            .iq3_s => 8,
            // T1-B2 (lane-b B2.15): type 9 = iq2_s.
            .iq2_s => 9,
            // T1-B3 (lane-b @a0e37f0): types 10-14 — mapping GEMM COMPLETO
            // 0..14 = todos los formatos con append (15/15).
            .iq4_nl => 10,
            .mxfp4 => 11,
            .iq3_xxs => 12,
            .iq2_xxs => 13,
            .iq2_xs => 14,
            // T1-B4 (lane-b @72cd4fb): case 15 = tq2_0.
            .tq2_0 => 15,
            // T1-B5 (lane-b @608dfc8/b13b6c2): case 16 = iq1_m — mapping
            // GEMM COMPLETO 0..16 = TODOS los formatos con append (17/17).
            .iq1_m => 16,
            // B-a4 (lane-a): case 17 = iq1_s — completa el mapping GEMM 17/17
            // (27B-IQ1_S). Kernel VERDE en qgemmKernel (espejo val_iq1_s).
            .iq1_s => 17,
            // Q1_0/Q2_0: cases 18/19 en qgemmKernel.
            .q1_0 => 18,
            .q2_0 => 19,
            // iq4_xs: kernel separado (iq4xsGemmM1Dp4aKernel), no parte de qgemmLinear.
            .iq4_xs => null,
            else => null,
        };
    }

    fn quantSsmEnabled() bool {
        return layer_kernels.quantPath() and !debugz.dbg.no_q4_ssm;
    }

    pub fn warmupGpuWeights(self: *SsmLayer) !void {
        const p = self.params;
        if (!quantSsmEnabled() or qgemmTypeFor(self.w_qkv.dtype()) == null) {
            try self.ensureScratchFilled();
            const qkv_dim = p.qkvDim();
            var w_qkv_shape = [_]usize{ qkv_dim, p.n_embd };
            var w_qkv_strides = [_]usize{ p.n_embd, 1 };
            _ = try self.matmul_engine.projectionDevicePtr(Tensor(f32){ .data = self.scratch_qkv, .shape = &w_qkv_shape, .strides = &w_qkv_strides, .offset = 0, .allocator = null, .owns_data = false });
            var w_z_shape = [_]usize{ p.d_inner, p.n_embd };
            var w_z_strides = [_]usize{ p.n_embd, 1 };
            _ = try self.matmul_engine.projectionDevicePtr(Tensor(f32){ .data = self.scratch_z, .shape = &w_z_shape, .strides = &w_z_strides, .offset = 0, .allocator = null, .owns_data = false });
        }
        if (!quantSsmEnabled() or qgemmTypeFor(self.w_out.dtype()) == null) {
            try self.ensureScratchFilled();
            var w_out_shape = [_]usize{ p.n_embd, p.d_inner };
            var w_out_strides = [_]usize{ p.d_inner, 1 };
            _ = try self.matmul_engine.projectionDevicePtr(Tensor(f32){ .data = self.scratch_out, .shape = &w_out_shape, .strides = &w_out_strides, .offset = 0, .allocator = null, .owns_data = false });
        }
        _ = try self.matmul_engine.projectionDevicePtr(self.w_beta);
        _ = try self.matmul_engine.projectionDevicePtr(self.w_alpha);
    }

    pub fn forwardGPU(
        self: *SsmLayer,
        lk: *layer_kernels.LayerKernels,
        x: cublas.GpuTensor(f32),
        out: *cublas.GpuTensor(f32),
        n: usize,
    ) !void {
        // §5.1 SEAM: dispatcher. prefill (n>1) → forwardPrefillChunked
        // (Dev C, §5.2). decode (n==1) → forwardGPUDecode — el camino
        // per-token donde §5.1 fundirá 6-7 kernels en UNO con el
        // S_state 64 KB en registros (unsloth §3.1).
        if (n > 1) {
            try self.forwardPrefillChunked(lk, x, out, n);
            return;
        }
        try self.forwardGPUDecode(lk, x, out, n);
    }

    /// §5.1: camino decode (n==1) per-token. Siguiente paso: reemplazar
    /// la cadena de 6-7 kernels con UN lanzamiento del fused GDN con
    /// S_state en registros (unsloth §3.1: grid (heads, 1, dim/4),
    /// block (32, 4, 1), decay+matmul+normas fused).
    pub fn forwardGPUDecode(
        self: *SsmLayer,
        lk: *layer_kernels.LayerKernels,
        x: cublas.GpuTensor(f32),
        out: *cublas.GpuTensor(f32),
        n: usize,
    ) !void {
        // PERF_SSM (lane-f F1): desglose por-etapa de la capa SSM en decode.
        // Requiere NOGRAPH (eventos CUDA por launch romperían el capture);
        // gated por PERF_STAGE → coste cero en producción. Acumula en
        // ssm_stage_ns por capa y reporta al deinit del SsmLayer.
        const perf_ssm = debugz.dbg.perf_stage and debugz.dbg.no_graph;
        var ev0: cudaz.CUevent = undefined;
        var ev1: cudaz.CUevent = undefined;
        if (perf_ssm) {
            ev0 = try cudaz.cuEventCreate(0);
            ev1 = try cudaz.cuEventCreate(0);
        }
        defer if (perf_ssm) {
            cudaz.cuEventDestroy(ev0);
            cudaz.cuEventDestroy(ev1);
        };

        // STUDY §5.1: camino decode (n==1) per-token. Siguiente paso: reemplazar
        // la cadena de 6-7 kernels con UN lanzamiento del fused GDN con
        // S_state en registros (unsloth §3.1: grid (heads, 1, dim/4),
        // block (32, 4, 1), decay+matmul+normas fused).
        const p = self.params;
        const qkv_dim = p.qkvDim();
        const key_dim = p.keyDim();
        const n_k_heads = p.n_group;
        const n_v_heads = p.dt_rank;
        const head_v_dim = p.d_state;
        const d_inner = p.d_inner;
        const dt_rank = p.dt_rank;

        try SsmLayer.ensureGpu(self);
        const g = &self.gpu.?;
        try g.ensureN(n);
        try cudaz.cuStreamSynchronize(lk.stream); // DEBUG
        debugz.dbg.printLevel(.detail, "[ssm] capa {d}: forwardGPU start n={d}\n", .{ self.layer_idx, n });

        // Pesos qkv/z/out como Tensor(f32) sobre los scratch dequantizados.
        var w_qkv_shape = [_]usize{ qkv_dim, p.n_embd };
        var w_qkv_strides = [_]usize{ p.n_embd, 1 };
        const w_qkv32 = Tensor(f32){ .data = self.scratch_qkv, .shape = &w_qkv_shape, .strides = &w_qkv_strides, .offset = 0, .allocator = null, .owns_data = false };
        var w_z_shape = [_]usize{ d_inner, p.n_embd };
        var w_z_strides = [_]usize{ p.n_embd, 1 };
        const w_z32 = Tensor(f32){ .data = self.scratch_z, .shape = &w_z_shape, .strides = &w_z_strides, .offset = 0, .allocator = null, .owns_data = false };
        var w_out_shape = [_]usize{ p.n_embd, d_inner };
        var w_out_strides = [_]usize{ d_inner, 1 };
        const w_out32 = Tensor(f32){ .data = self.scratch_out, .shape = &w_out_shape, .strides = &w_out_strides, .offset = 0, .allocator = null, .owns_data = false };

        // T1 VRAM-spec: GEMM cuantizado para TODO dtype con kernel (q8_0 en
        // particular — ssm_out/attn_qkv del Q8_K_XL). Fallback f32 solo para
        // dtypes sin kernel (f16/f32).
        const qt_qkv = if (quantSsmEnabled()) qgemmTypeFor(self.w_qkv.dtype()) else null;
        const qt_z = if (quantSsmEnabled()) qgemmTypeFor(self.w_z.dtype()) else null;
        if (perf_ssm) try cudaz.cuEventRecord(ev0, lk.stream);
        try cudaz.cuStreamSynchronize(lk.stream); // DEBUG: catch previous stream error
        if (qt_qkv != null and qt_z != null) {
            debugz.dbg.printLevel(.detail, "[ssm] capa {d}: calling qgemmLinear qkv m={d} k={d} n={d}\n", .{ self.layer_idx, n, p.n_embd, qkv_dim });
            lk.qgemmLinear(self.allocator, x.ptr(), self.w_qkv.bytes, g.qkv.ptr(), n, p.n_embd, qkv_dim, qt_qkv.?) catch |err| {
                debugz.dbg.printLevel(.info, "[ssm] capa {d}: qgemmLinear qkv FAILED: {s}\n", .{ self.layer_idx, @errorName(err) });
                return err;
            };
            debugz.dbg.printLevel(.detail, "[ssm] capa {d}: qgemmLinear qkv OK\n", .{self.layer_idx});
            lk.qgemmLinear(self.allocator, x.ptr(), self.w_z.bytes, g.z.ptr(), n, p.n_embd, d_inner, qt_z.?) catch |err| {
                debugz.dbg.printLevel(.info, "[ssm] capa {d}: qgemmLinear z FAILED: {s}\n", .{ self.layer_idx, @errorName(err) });
                return err;
            };
            debugz.dbg.printLevel(.detail, "[ssm] capa {d}: qgemmLinear z OK\n", .{self.layer_idx});
        } else {
            try self.ensureScratchFilled();
            try self.matmul_engine.linearProjectionDevice(x, w_qkv32, &g.qkv, n, p.n_embd, qkv_dim);
            try self.matmul_engine.linearProjectionDevice(x, w_z32, &g.z, n, p.n_embd, d_inner);
        }
        if (perf_ssm) {
            try cudaz.cuEventRecord(ev1, lk.stream);
            // ElapsedTime exige ambos eventos signaled — el stream es async y
            // el host llegaba antes (ERROR_NOT_READY → CudaError). Sync contra
            // el propio evento END antes de leer el delta.
            try cudaz.cuEventSynchronize(ev1);
            var ms: f32 = 0;
            try cudaz.cuEventElapsedTime(&ms, ev0, ev1);
            ssm_stage_ns[0] += @intFromFloat(@as(f64, ms) * std.time.ns_per_ms);
        }
        // beta/alpha: proyección f32 del mismo x + sigmoid(beta) + gateCompute en un
        // solo kernel (reemplaza 2 cublas SGEMM + sigmoidGate). Los pesos se suben
        // a GPU de forma cacheada vía projectionDevicePtr.
        const w_beta_dev = try self.matmul_engine.projectionDevicePtr(self.w_beta);
        const w_alpha_dev = try self.matmul_engine.projectionDevicePtr(self.w_alpha);
        if (debugz.dbg.dump_prefill_layers) {
            try cudaz.cuStreamSynchronize(lk.stream);
            const abuf = try self.allocator.alloc(f32, dt_rank);
            defer self.allocator.free(abuf);
            try cudaz.cuMemcpyDtoH(@intFromPtr(abuf.ptr), @intFromPtr(g.d_ssm_a.dev_ptr), dt_rank * @sizeOf(f32));
            const dbuf = try self.allocator.alloc(f32, dt_rank);
            defer self.allocator.free(dbuf);
            try cudaz.cuMemcpyDtoH(@intFromPtr(dbuf.ptr), @intFromPtr(g.d_dt_bias.dev_ptr), dt_rank * @sizeOf(f32));
            debugz.dbg.print("[ssm] SSMA sum={d:.6} max={d:.6}\n", .{ debugz.sumAbsF32(abuf), debugz.maxAbsF32(abuf) });
            debugz.dbg.print("[ssm] DTBIAS sum={d:.6} max={d:.6}\n", .{ debugz.sumAbsF32(dbuf), debugz.maxAbsF32(dbuf) });
            const wab = try self.allocator.alloc(f32, dt_rank * p.n_embd);
            defer self.allocator.free(wab);
            try cudaz.cuMemcpyDtoH(@intFromPtr(wab.ptr), w_alpha_dev, dt_rank * p.n_embd * @sizeOf(f32));
            debugz.dbg.print("[ssm] WALPHA sum={d:.6} max={d:.6}\n", .{ debugz.sumAbsF32(wab), debugz.maxAbsF32(wab) });
            const wbb = try self.allocator.alloc(f32, dt_rank * p.n_embd);
            defer self.allocator.free(wbb);
            try cudaz.cuMemcpyDtoH(@intFromPtr(wbb.ptr), w_beta_dev, dt_rank * p.n_embd * @sizeOf(f32));
            debugz.dbg.print("[ssm] WBETA sum={d:.6} max={d:.6}\n", .{ debugz.sumAbsF32(wbb), debugz.maxAbsF32(wbb) });
            debugz.dbg.print("[ssm] PTRS wbeta={*} walpha={*} wbdev={d} wadev={d}\n", .{ self.w_beta.data.ptr, self.w_alpha.data.ptr, w_beta_dev, w_alpha_dev });
        }
        if (perf_ssm) try cudaz.cuEventRecord(ev0, lk.stream);
        try lk.sigmoidGateProj(x.ptr(), w_beta_dev, w_alpha_dev, @intFromPtr(g.d_dt_bias.dev_ptr), @intFromPtr(g.d_ssm_a.dev_ptr), g.beta.ptr(), g.gate.ptr(), n, p.n_embd, dt_rank);
        if (perf_ssm) {
            try cudaz.cuEventRecord(ev1, lk.stream);
            // ElapsedTime exige ambos eventos signaled — el stream es async y
            // el host llegaba antes (ERROR_NOT_READY → CudaError). Sync contra
            // el propio evento END antes de leer el delta.
            try cudaz.cuEventSynchronize(ev1);
            var ms: f32 = 0;
            try cudaz.cuEventElapsedTime(&ms, ev0, ev1);
            ssm_stage_ns[1] += @intFromFloat(@as(f64, ms) * std.time.ns_per_ms);
        }

        // conv1d causal + silu leyendo conv_state/qkv directo (sin staging); el
        // estado desplazado se escribe en conv_in (scratch) y se copia de vuelta.
        if (perf_ssm) try cudaz.cuEventRecord(ev0, lk.stream);
        try lk.conv1dSilu(@intFromPtr(g.d_conv_state.dev_ptr), g.qkv.ptr(), @intFromPtr(g.d_conv1d.dev_ptr), g.conv_out.ptr(), g.conv_in.ptr(), n, qkv_dim, p.d_conv);
        try cudaz.cuMemcpyDtoDAsync(@intFromPtr(g.d_conv_state.dev_ptr), g.conv_in.ptr(), (p.d_conv - 1) * qkv_dim * @sizeOf(f32), lk.stream);
        if (perf_ssm) {
            try cudaz.cuEventRecord(ev1, lk.stream);
            // ElapsedTime exige ambos eventos signaled — el stream es async y
            // el host llegaba antes (ERROR_NOT_READY → CudaError). Sync contra
            // el propio evento END antes de leer el delta.
            try cudaz.cuEventSynchronize(ev1);
            var ms: f32 = 0;
            try cudaz.cuEventElapsedTime(&ms, ev0, ev1);
            ssm_stage_ns[2] += @intFromFloat(@as(f64, ms) * std.time.ns_per_ms);
        }

        // STUDY §5.1 FUSED: l2 + ΔNet(S en registros, port unsloth §3.1) +
        // rmsNorm·silu(z) en UN lanzamiento por (head, token) — sustituye
        // l2NormHeads + deltaNet{,Warp}×n + rmsNormGateMul (3+ kernels → 1).
        // Ahora DEFAULT para configs compatibles (dim%32==0, dim<=128,
        // n_v_heads==n_k_heads). Fuera de ellas cae a la cadena separada.
        if (perf_ssm) try cudaz.cuEventRecord(ev0, lk.stream);
        if (head_v_dim % 32 == 0 and head_v_dim <= 128 and n_v_heads == n_k_heads) {
            debugz.dbg.printLevel(.detail, "[ssm] capa {d}: deltaNetFused (l2+ΔNet+rms en 1 launch)\n", .{self.layer_idx});
            try lk.deltaNetFused(
                g.conv_out.ptr(),
                g.gate.ptr(),
                g.beta.ptr(),
                g.z.ptr(),
                @intFromPtr(g.d_ssm_norm.dev_ptr),
                g.attn_out.ptr(),
                @intFromPtr(g.d_s_state.dev_ptr),
                n,
                qkv_dim,
                key_dim,
                n_k_heads,
                n_v_heads,
                head_v_dim,
                dt_rank,
                p.rms_eps,
            );
        } else {
            try lk.l2NormHeads(g.conv_out.ptr(), n, qkv_dim, key_dim, n_k_heads, head_v_dim, p.rms_eps);

            // DeltaNet: recurrencia secuencial por token (estado persiste en d_s_state).
            if (debugz.dbg.dump_prefill_layers) {
                try cudaz.cuStreamSynchronize(lk.stream);
                const qbuf = try self.allocator.alloc(f32, n * qkv_dim);
                defer self.allocator.free(qbuf);
                try cudaz.cuMemcpyDtoH(@intFromPtr(qbuf.ptr), g.qkv.ptr(), n * qkv_dim * @sizeOf(f32));
                const cbuf = try self.allocator.alloc(f32, n * qkv_dim);
                defer self.allocator.free(cbuf);
                try cudaz.cuMemcpyDtoH(@intFromPtr(cbuf.ptr), g.conv_out.ptr(), n * qkv_dim * @sizeOf(f32));
                const gbuf = try self.allocator.alloc(f32, n * dt_rank);
                defer self.allocator.free(gbuf);
                try cudaz.cuMemcpyDtoH(@intFromPtr(gbuf.ptr), g.gate.ptr(), n * dt_rank * @sizeOf(f32));
                const bbuf = try self.allocator.alloc(f32, n * dt_rank);
                defer self.allocator.free(bbuf);
                try cudaz.cuMemcpyDtoH(@intFromPtr(bbuf.ptr), g.beta.ptr(), n * dt_rank * @sizeOf(f32));
                debugz.dbg.print("[ssm] QKV sum={d:.6} max={d:.6}\n", .{ debugz.sumAbsF32(qbuf), debugz.maxAbsF32(qbuf) });
                debugz.dbg.print("[ssm] CONVOUT sum={d:.6} max={d:.6}\n", .{ debugz.sumAbsF32(cbuf), debugz.maxAbsF32(cbuf) });
                debugz.dbg.print("[ssm] GATE sum={d:.6} max={d:.6}\n", .{ debugz.sumAbsF32(gbuf), debugz.maxAbsF32(gbuf) });
                debugz.dbg.print("[ssm] BETA sum={d:.6} max={d:.6}\n", .{ debugz.sumAbsF32(bbuf), debugz.maxAbsF32(bbuf) });
            }
            var t: usize = 0;
            while (t < n) : (t += 1) {
                const co = g.conv_out.ptr() + t * qkv_dim * @sizeOf(f32);
                const ga = g.gate.ptr() + t * dt_rank * @sizeOf(f32);
                const be = g.beta.ptr() + t * dt_rank * @sizeOf(f32);
                const ao = g.attn_out.ptr() + t * d_inner * @sizeOf(f32);
                // STUDY §5.6: warp-shuffle sin barreras. OPT-IN: DNWARP=1 → warp;
                // unset → clásico (árbol seguro por defecto — nunca decode-unsafe
                // con WIP experimental).
                if (head_v_dim % 32 == 0 and std.c.getenv("DNWARP") != null) {
                    try lk.deltaNetWarp(co, ga, be, ao, @intFromPtr(g.d_s_state.dev_ptr), 1, qkv_dim, key_dim, n_k_heads, n_v_heads, head_v_dim, dt_rank, p.rms_eps);
                } else {
                    try lk.deltaNet(co, ga, be, ao, @intFromPtr(g.d_s_state.dev_ptr), 1, qkv_dim, key_dim, n_k_heads, n_v_heads, head_v_dim, dt_rank, p.rms_eps);
                }
            }

            try lk.rmsNormGateMul(g.attn_out.ptr(), g.z.ptr(), @intFromPtr(g.d_ssm_norm.dev_ptr), n, d_inner, n_v_heads, head_v_dim, p.rms_eps);
        }
        if (perf_ssm) {
            try cudaz.cuEventRecord(ev1, lk.stream);
            // ElapsedTime exige ambos eventos signaled — el stream es async y
            // el host llegaba antes (ERROR_NOT_READY → CudaError). Sync contra
            // el propio evento END antes de leer el delta.
            try cudaz.cuEventSynchronize(ev1);
            var ms: f32 = 0;
            try cudaz.cuEventElapsedTime(&ms, ev0, ev1);
            ssm_stage_ns[3] += @intFromFloat(@as(f64, ms) * std.time.ns_per_ms);
        }
        // T1 VRAM-spec: mismo tratamiento cuantizado para w_out (q8_0/q4_k/
        // q6_k/... según modelo; antes solo q4_0/q5_k).
        const qt_out = if (quantSsmEnabled()) qgemmTypeFor(self.w_out.dtype()) else null;
        if (perf_ssm) try cudaz.cuEventRecord(ev0, lk.stream);
        if (qt_out) |qt| {
            try lk.qgemmLinear(self.allocator, g.attn_out.ptr(), self.w_out.bytes, out.ptr(), n, d_inner, p.n_embd, qt);
        } else {
            try self.ensureScratchFilled();
            try self.matmul_engine.linearProjectionDevice(g.attn_out, w_out32, out, n, d_inner, p.n_embd);
        }
        if (perf_ssm) {
            try cudaz.cuEventRecord(ev1, lk.stream);
            // ElapsedTime exige ambos eventos signaled — el stream es async y
            // el host llegaba antes (ERROR_NOT_READY → CudaError). Sync contra
            // el propio evento END antes de leer el delta.
            try cudaz.cuEventSynchronize(ev1);
            var ms: f32 = 0;
            try cudaz.cuEventElapsedTime(&ms, ev0, ev1);
            ssm_stage_ns[4] += @intFromFloat(@as(f64, ms) * std.time.ns_per_ms);
            ssm_stage_tokens += n;
        }
    }

    /// Copia el estado recurrente host (s_state, conv_state) — resultante del prefill
    /// CPU — a los buffers GPU persistentes. Necesario porque decode corre por GPU.
    pub fn seedGpuFromHost(self: *SsmLayer) !void {
        debugz.dbg.printLevel(.detail, "[ssm] capa {d}: seedGpuFromHost s_state.len={d} conv.len={d}\n", .{ self.layer_idx, self.s_state.len, self.conv_state.len });
        try SsmLayer.ensureGpu(self);
        const g = &self.gpu.?;
        try cudaz.cuMemcpyHtoD(@intFromPtr(g.d_s_state.dev_ptr), @intFromPtr(self.s_state.ptr), self.s_state.len * @sizeOf(f32));
        try cudaz.cuMemcpyHtoD(@intFromPtr(g.d_conv_state.dev_ptr), @intFromPtr(self.conv_state.ptr), self.conv_state.len * @sizeOf(f32));
        debugz.dbg.printLevel(.detail, "[ssm] capa {d}: seedGpuFromHost OK\n", .{self.layer_idx});
    }

    /// STUDY §5.2: prefill chunked batched ΔNet. Procesa K tokens por v-head en
    /// una lanzada (K=64 default, K=128 vía PREFILLK=128), manteniendo el estado
    /// S_v×S_v en registros a lo largo del bucle de K tokens. Los tokens restantes
    /// (n % K != 0) caen al camino per-token clásico. kda=false para Qwen3.5
    /// (decay escalar por cabeza). OPT-IN: PREFILL_chunk=1 fuerza el camino
    /// chunked incluso con n==1 (para tests de paridad); unset deja n==1 en clásico.
    pub fn forwardPrefillChunked(
        self: *SsmLayer,
        lk: *layer_kernels.LayerKernels,
        x: cublas.GpuTensor(f32),
        out: *cublas.GpuTensor(f32),
        n: usize,
    ) !void {
        const p = self.params;
        const qkv_dim = p.qkvDim();
        const key_dim = p.keyDim();
        const n_k_heads = p.n_group;
        const n_v_heads = p.dt_rank;
        const head_v_dim = p.d_state;
        const d_inner = p.d_inner;
        const dt_rank = p.dt_rank;
        const S_v = head_v_dim;

        try SsmLayer.ensureGpu(self);
        const g = &self.gpu.?;
        try g.ensureN(n);

        var w_qkv_shape = [_]usize{ qkv_dim, p.n_embd };
        var w_qkv_strides = [_]usize{ p.n_embd, 1 };
        const w_qkv32 = Tensor(f32){ .data = self.scratch_qkv, .shape = &w_qkv_shape, .strides = &w_qkv_strides, .offset = 0, .allocator = null, .owns_data = false };
        var w_z_shape = [_]usize{ d_inner, p.n_embd };
        var w_z_strides = [_]usize{ p.n_embd, 1 };
        const w_z32 = Tensor(f32){ .data = self.scratch_z, .shape = &w_z_shape, .strides = &w_z_strides, .offset = 0, .allocator = null, .owns_data = false };
        var w_out_shape = [_]usize{ p.n_embd, d_inner };
        var w_out_strides = [_]usize{ d_inner, 1 };
        const w_out32 = Tensor(f32){ .data = self.scratch_out, .shape = &w_out_shape, .strides = &w_out_strides, .offset = 0, .allocator = null, .owns_data = false };

        // 1. Proyección qkv + z (batched GEMM).
        const qt_qkv = if (quantSsmEnabled()) qgemmTypeFor(self.w_qkv.dtype()) else null;
        const qt_z = if (quantSsmEnabled()) qgemmTypeFor(self.w_z.dtype()) else null;
        if (qt_qkv != null and qt_z != null) {
            try lk.qgemmLinear(self.allocator, x.ptr(), self.w_qkv.bytes, g.qkv.ptr(), n, p.n_embd, qkv_dim, qt_qkv.?);
            try lk.qgemmLinear(self.allocator, x.ptr(), self.w_z.bytes, g.z.ptr(), n, p.n_embd, d_inner, qt_z.?);
        } else {
            try self.ensureScratchFilled();
            try self.matmul_engine.linearProjectionDevice(x, w_qkv32, &g.qkv, n, p.n_embd, qkv_dim);
            try self.matmul_engine.linearProjectionDevice(x, w_z32, &g.z, n, p.n_embd, d_inner);
        }

        // 2. beta/alpha proyección + sigmoid/gate.
        const w_beta_dev = try self.matmul_engine.projectionDevicePtr(self.w_beta);
        const w_alpha_dev = try self.matmul_engine.projectionDevicePtr(self.w_alpha);
        try lk.sigmoidGateProj(x.ptr(), w_beta_dev, w_alpha_dev, @intFromPtr(g.d_dt_bias.dev_ptr), @intFromPtr(g.d_ssm_a.dev_ptr), g.beta.ptr(), g.gate.ptr(), n, p.n_embd, dt_rank);

        // 3. conv1d causal + silu + l2 (conv1dSiluL2 wrapper, default-on).
        try lk.conv1dSiluL2(@intFromPtr(g.d_conv_state.dev_ptr), g.qkv.ptr(), @intFromPtr(g.d_conv1d.dev_ptr), g.conv_out.ptr(), g.conv_in.ptr(), n, qkv_dim, p.d_conv, key_dim, n_k_heads, n_v_heads, head_v_dim, p.rms_eps);
        try cudaz.cuMemcpyDtoDAsync(@intFromPtr(g.d_conv_state.dev_ptr), g.conv_in.ptr(), (p.d_conv - 1) * qkv_dim * @sizeOf(f32), lk.stream);

        // 4. ΔNet chunked prefill.
        // 1.11 (lane-c): DEFAULT fused CH ubatch-wide — 1 launch/capa
        // para el chunk completo del prefill (ref delta-net-base.cpp:373-447
        // build_delta_net_fused con K=n_tokens; antes 8 launches K=64 + el
        // branch de cola que caía a per-token: n=100 ⇒ 1×K64 + 36 per-token).
        // El kernel procesa n_tokens runtime (v5) — la cola parcial va en el
        // MISMO launch (sin OOB: fix §5.2 n_tokens reales). Opt-out PREFILLCHUNKED
        // (p.ej. PREFILLCHUNKED=1 fuerza el camino chunked; unset = WY default).
        // 1.4 (lane-b): PREFILLWY=1 era el opt-in del camino WY batched; ahora
        // es DEFAULT. Opt-out PREFILLWY=0.
        var K: usize = 512;
        if (std.c.getenv("PREFILLK")) |raw| {
            // parse [:0]u8 manually (Zig 0.16 doesn't allow .len on sentinel slices)
            var v: usize = 0;
            var i: usize = 0;
            while (raw[i] != 0) : (i += 1) {
                const digit = raw[i] - '0';
                if (digit > 9) {
                    v = 512;
                    break;
                } // invalid char → default
                v = v * 10 + digit;
            }
            if (v != 0) K = v;
        }
        const use_chunked = blk: {
            const raw = std.c.getenv("PREFILLCHUNKED");
            break :blk raw != null and raw.?[0] == '1' and raw.?[1] == 0;
        };
        const use_wy_default = blk: {
            const raw = std.c.getenv("PREFILLWY");
            break :blk raw == null or raw.?[0] == '1';
        };
        const effective_wy = !use_chunked and use_wy_default;
        const kda = false; // Qwen3.5: decay escalar por cabeza
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(S_v)));
        const q_off: c_int = 0;
        const k_off: c_int = @intCast(key_dim);
        const v_off: c_int = @intCast(2 * key_dim);
        const qkv_stride: c_int = @intCast(qkv_dim);
        const d_inner_c: c_int = @intCast(d_inner);
        const n_v_heads_c: c_int = @intCast(n_v_heads);
        const n_k_heads_c: c_int = @intCast(n_k_heads);
        const hv_dim_c: c_int = @intCast(head_v_dim);
        const dt_stride: c_int = @intCast(dt_rank);

        var t: usize = 0;
        if (effective_wy) {
            // 1.4 (lane-b): camino WY — 2 launches para TODO el prefill (K1
            // solve batched sobre n_chunks×n_v_heads, K2 state por columnas).
            // El kernel trocea n en chunks internos de 64; estado INOUT igual
            // que v5. Breadcrumb de launch para el gate de paridad.
            debugz.dbg.printLevel(.trace, "[prefill_wy] layer={d} n={d} chunks={d}\n", .{ self.layer_idx, n, (n + 63) / 64 });
            if (debugz.dbg.dump_kv) {
                // 1.4 (lane-f): dump binario de las entradas reales layer N
                // para repro aislado del oráculo — vía (1) del debug E2E de
                // Dev-B (HANDOFFS 2026-09-09 23:3x). Gate: DUMPKV=1 + trace.
                const nq = n * qkv_dim;
                const ng = n * dt_rank;
                const conv_host = try self.allocator.alloc(f32, nq);
                defer self.allocator.free(conv_host);
                const gate_host = try self.allocator.alloc(f32, ng);
                defer self.allocator.free(gate_host);
                const beta_host = try self.allocator.alloc(f32, ng);
                defer self.allocator.free(beta_host);
                const st_host = try self.allocator.alloc(f32, n_v_heads * head_v_dim * head_v_dim);
                defer self.allocator.free(st_host);
                const out_host = try self.allocator.alloc(f32, n * d_inner);
                defer self.allocator.free(out_host);
                try cudaz.cuMemcpyDtoH(@intFromPtr(conv_host.ptr), g.conv_out.ptr(), nq * @sizeOf(f32));
                try cudaz.cuMemcpyDtoH(@intFromPtr(gate_host.ptr), g.gate.ptr(), ng * @sizeOf(f32));
                try cudaz.cuMemcpyDtoH(@intFromPtr(beta_host.ptr), g.beta.ptr(), ng * @sizeOf(f32));
                try cudaz.cuMemcpyDtoH(@intFromPtr(st_host.ptr), @intFromPtr(g.d_s_state.dev_ptr), n_v_heads * head_v_dim * head_v_dim * @sizeOf(f32));
                try cudaz.cuMemcpyDtoH(@intFromPtr(out_host.ptr), g.attn_out.ptr(), n * d_inner * @sizeOf(f32));
                var path_buf: [128]u8 = undefined;
                const path = try std.fmt.bufPrint(&path_buf, "/tmp/opencode/wydump_l{d}.bin", .{self.layer_idx});
                if (std.Io.Dir.cwd().createFile(std.Io.Threaded.global_single_threaded.io(), path, .{ .truncate = true })) |f| {
                    defer f.close(std.Io.Threaded.global_single_threaded.io());
                    var buf: [4096]u8 = undefined;
                    var w = f.writer(std.Io.Threaded.global_single_threaded.io(), &buf);
                    const wr = &w.interface;
                    try wr.print("WYDUMP n={d} qkv_dim={d} dt_rank={d} nvh={d} nkh={d} hvd={d} d_inner={d}\n", .{ n, qkv_dim, dt_rank, n_v_heads, n_k_heads, head_v_dim, d_inner });
                    try wr.writeAll(std.mem.sliceAsBytes(conv_host));
                    try wr.writeAll(std.mem.sliceAsBytes(gate_host));
                    try wr.writeAll(std.mem.sliceAsBytes(beta_host));
                    try wr.writeAll(std.mem.sliceAsBytes(st_host));
                    try wr.writeAll(std.mem.sliceAsBytes(out_host));
                    try wr.flush();
                    debugz.dbg.printLevel(.info, "[prefill_wy] dump layer={d} -> {s}\n", .{ self.layer_idx, path });
                } else |_| {}
            }
            try lk.prefillWY(
                g.conv_out.ptr(),
                q_off,
                k_off,
                v_off,
                qkv_stride,
                g.gate.ptr(),
                dt_stride,
                g.beta.ptr(),
                @intFromPtr(g.d_s_state.dev_ptr),
                g.attn_out.ptr(),
                d_inner_c,
                scale,
                n_v_heads_c,
                n_k_heads_c,
                hv_dim_c,
                @intCast(n),
            );
            if (debugz.dbg.dump_kv) {
                // 1.4 (lane-f): attn_out POST-kernel del run E2E — compara
                // con el oráculo en el repro aislado (si difiere, la
                // divergencia es async/wiring; si coincide, es posterior).
                const out_post = try self.allocator.alloc(f32, n * d_inner);
                defer self.allocator.free(out_post);
                try cudaz.cuStreamSynchronize(lk.stream);
                try cudaz.cuMemcpyDtoH(@intFromPtr(out_post.ptr), g.attn_out.ptr(), n * d_inner * @sizeOf(f32));
                var path_buf2: [128]u8 = undefined;
                const path2 = try std.fmt.bufPrint(&path_buf2, "/tmp/opencode/wydump_l{d}.post", .{self.layer_idx});
                if (std.Io.Dir.cwd().createFile(std.Io.Threaded.global_single_threaded.io(), path2, .{ .truncate = true })) |f2| {
                    defer f2.close(std.Io.Threaded.global_single_threaded.io());
                    var buf2: [4096]u8 = undefined;
                    var w2 = f2.writer(std.Io.Threaded.global_single_threaded.io(), &buf2);
                    const wr2 = &w2.interface;
                    try wr2.writeAll(std.mem.sliceAsBytes(out_post));
                    try wr2.flush();
                } else |_| {}
            }
        } else while (t < n) : (t += K) {
            // 1.11 (lane-c): la cola parcial (n_tokens < K) va en el MISMO
            // launch — el kernel v5 procesa n_tokens runtime sin OOB (fix
            // §5.2). El branch anterior la mandaba al camino per-token:
            // n=100 con K=64 ⇒ 1 chunk + 36 launches per-token (la cola
            // dominaba el coste). Per-token puro sigue disponible con
            // PREFILLK=1.
            const co_ptr = g.conv_out.ptr() + t * qkv_dim * @sizeOf(f32);
            const ao_ptr = g.attn_out.ptr() + t * d_inner * @sizeOf(f32);
            const g_ptr = g.gate.ptr() + t * dt_rank * @sizeOf(f32);
            const b_ptr = g.beta.ptr() + t * dt_rank * @sizeOf(f32);
            // Fix §5.2 (Dev C): tokens REALES del chunk. El kernel recibe
            // n_tokens para no leer/escribir OOB cuando el chunk es parcial
            // (buffers dimensionados a n, no a K).
            const k_actual: usize = @min(K, n - t);
            const t_start_c: c_int = @intCast(t);

            debugz.dbg.printLevel(.trace, "[prefill_chunk] layer={d} t={d} K={d} n_tok={d}\n", .{ self.layer_idx, t, K, k_actual });
            try lk.prefillDeltaNetChunk(
                co_ptr,
                q_off,
                k_off,
                v_off,
                qkv_stride,
                g_ptr,
                dt_stride,
                b_ptr,
                @intFromPtr(g.d_s_state.dev_ptr),
                ao_ptr,
                d_inner_c,
                t_start_c,
                scale,
                n_v_heads_c,
                n_k_heads_c,
                hv_dim_c,
                @intCast(K),
                kda,
                @intCast(k_actual),
            );

            if (debugz.dbg.dump_prefill_layers) {
                try cudaz.cuStreamSynchronize(lk.stream);
                debugz.dbg.printLevel(.info, "[prefill_chunk] chunk @ t={d} K={d} kda={}\n", .{ t, K, kda });
            }
        }

        // 5. rmsNormGateMul + out projection.
        try lk.rmsNormGateMul(g.attn_out.ptr(), g.z.ptr(), @intFromPtr(g.d_ssm_norm.dev_ptr), n, d_inner, n_v_heads, head_v_dim, p.rms_eps);
        const qt_out = if (quantSsmEnabled()) qgemmTypeFor(self.w_out.dtype()) else null;
        if (qt_out) |qt| {
            try lk.qgemmLinear(self.allocator, g.attn_out.ptr(), self.w_out.bytes, out.ptr(), n, d_inner, p.n_embd, qt);
        } else {
            try self.ensureScratchFilled();
            try self.matmul_engine.linearProjectionDevice(g.attn_out, w_out32, out, n, d_inner, p.n_embd);
        }
    }
};

fn loadQuantWeight(g: *const gguf.GgufFile, prefix: []const u8, name: []const u8) !QuantWeight {
    const full = try std.fmt.allocPrint(std.heap.page_allocator, "{s}{s}", .{ prefix, name });
    defer std.heap.page_allocator.free(full);
    const info = g.getTensor(full) orelse return SsmError.WeightFileNotFound;
    return QuantWeight.init(info, g.tensorData(info));
}

fn loadGgufF32(
    allocator: std.mem.Allocator,
    g: *const gguf.GgufFile,
    prefix: []const u8,
    name: []const u8,
    transpose: bool,
) !Tensor(f32) {
    const full = try std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, name });
    defer allocator.free(full);
    const info = g.getTensor(full) orelse return SsmError.WeightFileNotFound;
    const numel: usize = @intCast(info.numel());

    const f32buf = try allocator.alloc(f32, numel);
    defer allocator.free(f32buf);
    try gguf.dequantTensor(info, g.tensorData(info), f32buf);

    var out_dim: usize = 1;
    var in_dim: usize = 1;
    var tensor: Tensor(f32) = undefined;
    if (info.n_dims >= 2) {
        // GGUF guarda [in, out]; las capas lineales esperan [out, in] → transponer.
        in_dim = @intCast(info.dims[0]);
        out_dim = @intCast(info.dims[1]);
        tensor = try Tensor(f32).initUninitialized(allocator, &.{ out_dim, in_dim });
        if (transpose) {
            for (0..in_dim) |r| {
                for (0..out_dim) |c| {
                    tensor.data[c * in_dim + r] = f32buf[r + c * in_dim];
                }
            }
        } else {
            @memcpy(tensor.data, f32buf);
        }
    } else {
        tensor = try Tensor(f32).initUninitialized(allocator, &.{numel});
        @memcpy(tensor.data, f32buf);
    }
    return tensor;
}

// ─── Tests ───

fn approx(a: f32, b: f32, tol: f32) bool {
    return @abs(a - b) <= tol;
}

/// Construye un SsmLayer pequeño con pesos controlados.
/// X = [1,1,1] → linearProjection fila j = fila j de la matriz de pesos.
const TestDims = struct {
    n_embd: usize = 3,
    d_inner: usize = 4,
    d_state: usize = 2,
    dt_rank: usize = 2,
    n_group: usize = 1,
    d_conv: usize = 1,
};

const test_dims = TestDims{};

/// Fixture: los pesos grandes (QuantWeight) viven en buffers de bytes f32
/// que el test posee (en producción los posee el GgufFile mmap). Los
/// TensorInfo se heap-alocan para que los punteros del QuantWeight no
/// queden colgando al devolver la fixture.
const TestFixture = struct {
    allocator: std.mem.Allocator,
    layer: SsmLayer,
    qkv_bytes: []u8,
    z_bytes: []u8,
    out_bytes: []u8,
    qkv_info: *gguf.TensorInfo,
    z_info: *gguf.TensorInfo,
    out_info: *gguf.TensorInfo,

    fn deinit(self: *TestFixture) void {
        self.layer.deinit();
        self.allocator.free(self.qkv_bytes);
        self.allocator.free(self.z_bytes);
        self.allocator.free(self.out_bytes);
        self.allocator.destroy(self.qkv_info);
        self.allocator.destroy(self.z_info);
        self.allocator.destroy(self.out_info);
    }
};

fn makeF32Weight(
    allocator: std.mem.Allocator,
    out_dim: usize,
    in_dim: usize,
    values: []const f32,
    info: *gguf.TensorInfo,
    bytes: *[]u8,
) !QuantWeight {
    // `values` es la matriz [out, in] (W[j][i]). El GGUF la guarda transpuesta
    // [in, out]; dequantToF16Transposed espera ese layout.
    info.* = gguf.TensorInfo{
        .name = "test",
        .n_dims = 2,
        .dims = .{ in_dim, out_dim, 0, 0 },
        .dtype = .f32,
        .offset = 0,
    };
    bytes.* = try allocator.alloc(u8, values.len * 4);
    const fbytes = std.mem.bytesAsSlice(f32, bytes.*);
    // GGUF: dim0 (in_dim) contiguo → elemento (i, j) en i + j*in_dim.
    for (0..in_dim) |i| {
        for (0..out_dim) |j| {
            fbytes[i + j * in_dim] = values[j * in_dim + i];
        }
    }
    return QuantWeight.init(info, bytes.*);
}

fn buildTestLayer(allocator: std.mem.Allocator) !TestFixture {
    const p = SsmParams{
        .n_embd = test_dims.n_embd,
        .d_inner = test_dims.d_inner,
        .d_state = test_dims.d_state,
        .dt_rank = test_dims.dt_rank,
        .n_group = test_dims.n_group,
        .d_conv = test_dims.d_conv,
    };
    var layer = try SsmLayer.init(allocator, 0, p, .auto);
    errdefer layer.deinit();

    const qkv_info = try allocator.create(gguf.TensorInfo);
    const z_info = try allocator.create(gguf.TensorInfo);
    const out_info = try allocator.create(gguf.TensorInfo);

    // w_qkv [8,3]: fila j = [target_j, 0, 0]
    const qkv_target = [_]f32{ 0.5, -0.5, 0.7, 0.3, 0.2, 0.8, 0.4, -0.6 };
    var qkv_vals: [24]f32 = undefined;
    for (0..8) |j| {
        qkv_vals[j * 3 + 0] = qkv_target[j];
        qkv_vals[j * 3 + 1] = 0;
        qkv_vals[j * 3 + 2] = 0;
    }
    var qkv_bytes: []u8 = undefined;
    layer.w_qkv = try makeF32Weight(allocator, 8, 3, &qkv_vals, qkv_info, &qkv_bytes);

    // w_z [4,3]
    const z_target = [_]f32{ 0.1, 0.2, -0.1, 0.3 };
    var z_vals: [12]f32 = undefined;
    for (0..4) |j| {
        z_vals[j * 3 + 0] = z_target[j];
        z_vals[j * 3 + 1] = 0;
        z_vals[j * 3 + 2] = 0;
    }
    var z_bytes: []u8 = undefined;
    layer.w_z = try makeF32Weight(allocator, 4, 3, &z_vals, z_info, &z_bytes);

    // w_beta [2,3]
    const beta_target = [_]f32{ 0.5, -0.5 };
    for (0..2) |j| {
        layer.w_beta.data[j * 3 + 0] = beta_target[j];
        layer.w_beta.data[j * 3 + 1] = 0;
        layer.w_beta.data[j * 3 + 2] = 0;
    }
    // w_alpha [2,3]
    const alpha_target = [_]f32{ 0.4, -0.2 };
    for (0..2) |j| {
        layer.w_alpha.data[j * 3 + 0] = alpha_target[j];
        layer.w_alpha.data[j * 3 + 1] = 0;
        layer.w_alpha.data[j * 3 + 2] = 0;
    }
    // dt_bias, ssm_a [2]
    layer.dt_bias.data[0] = 0.1;
    layer.dt_bias.data[1] = 0.3;
    layer.ssm_a.data[0] = 0.8;
    layer.ssm_a.data[1] = 1.2;
    // conv1d [8,1] = 1.0 (d_conv=1 → identidad)
    for (layer.conv1d.data) |*v| v.* = 1.0;
    // ssm_norm [2] = 1.0
    for (layer.ssm_norm.data) |*v| v.* = 1.0;
    // w_out [3,4]: fila j = [e_j, 0, 0, 0]
    const out_target = [_]f32{ 1.0, 2.0, 3.0 };
    var out_vals: [12]f32 = undefined;
    for (0..3) |j| {
        out_vals[j * 4 + 0] = out_target[j];
        out_vals[j * 4 + 1] = 0;
        out_vals[j * 4 + 2] = 0;
        out_vals[j * 4 + 3] = 0;
    }
    var out_bytes: []u8 = undefined;
    layer.w_out = try makeF32Weight(allocator, 3, 4, &out_vals, out_info, &out_bytes);

    // forward lee los pesos grandes ya dequantizados desde scratch_* (como hace
    // loadWeightsFromGguf); el fixture debe rellenarlos o forward vería ceros.
    layer.w_qkv.dequantToF32Transposed(layer.scratch_qkv);
    layer.w_z.dequantToF32Transposed(layer.scratch_z);
    layer.w_out.dequantToF32Transposed(layer.scratch_out);

    return .{
        .allocator = allocator,
        .layer = layer,
        .qkv_bytes = qkv_bytes,
        .z_bytes = z_bytes,
        .out_bytes = out_bytes,
        .qkv_info = qkv_info,
        .z_info = z_info,
        .out_info = out_info,
    };
}

/// Igual que buildTestLayer pero con pesos densos (todas las columnas no
/// nulas) para que tokens distintos produzcan proyecciones distintas. Los
/// valores son arbitrarios; los tests de densidad usan checks de consistencia.
fn buildDenseLayer(allocator: std.mem.Allocator) !TestFixture {
    var fixture = try buildTestLayer(allocator);
    errdefer fixture.deinit();
    const layer = &fixture.layer;

    const dense = struct {
        fn val(row: usize, col: usize) f32 {
            return @as(f32, @floatFromInt(row + 1)) * 0.13 + @as(f32, @floatFromInt(col + 1)) * 0.07;
        }
    }.val;

    // reescribir los pesos grandes como f32 densos
    var qkv_vals: [24]f32 = undefined;
    var z_vals: [12]f32 = undefined;
    var out_vals: [12]f32 = undefined;
    for (0..8) |j| {
        for (0..3) |c| qkv_vals[j * 3 + c] = dense(j, c);
    }
    for (0..4) |j| {
        for (0..3) |c| z_vals[j * 3 + c] = dense(j, c) * 0.5;
    }
    for (0..3) |j| {
        for (0..4) |c| out_vals[j * 4 + c] = dense(j, c) * 0.9;
    }
    // liberar los buffers f32 del buildTestLayer antes de reasignar
    allocator.free(fixture.qkv_bytes);
    allocator.free(fixture.z_bytes);
    allocator.free(fixture.out_bytes);
    layer.w_qkv = try makeF32Weight(allocator, 8, 3, &qkv_vals, fixture.qkv_info, &fixture.qkv_bytes);
    layer.w_z = try makeF32Weight(allocator, 4, 3, &z_vals, fixture.z_info, &fixture.z_bytes);
    layer.w_out = try makeF32Weight(allocator, 3, 4, &out_vals, fixture.out_info, &fixture.out_bytes);
    layer.w_qkv.dequantToF32Transposed(layer.scratch_qkv);
    layer.w_z.dequantToF32Transposed(layer.scratch_z);
    layer.w_out.dequantToF32Transposed(layer.scratch_out);

    for (0..2) |j| {
        for (0..3) |c| {
            layer.w_beta.data[j * 3 + c] = dense(j, c) * 0.4;
            layer.w_alpha.data[j * 3 + c] = dense(j, c) * 0.3;
        }
    }
    layer.dt_bias.data[0] = 0.1;
    layer.dt_bias.data[1] = 0.3;
    layer.ssm_a.data[0] = 0.8;
    layer.ssm_a.data[1] = 1.2;
    for (layer.ssm_norm.data) |*v| v.* = 1.0;
    return fixture;
}

test "ssm forward single token hand-computed" {
    const allocator = std.testing.allocator;
    var fixture = try buildTestLayer(allocator);
    defer fixture.deinit();
    var layer = &fixture.layer;

    var x = try Tensor(f32).alloc(allocator, &.{ 1, test_dims.n_embd });
    defer x.deinit();
    x.data[0] = 1.0;
    x.data[1] = 1.0;
    x.data[2] = 1.0;

    var out = try Tensor(f32).alloc(allocator, &.{ 1, test_dims.n_embd });
    defer out.deinit();

    try layer.forward(x, &out, 1);

    // Esperado (calculado a mano con los valores de arriba)
    try std.testing.expect(approx(@floatCast(out.data[0]), 0.0145, 1e-3));
    try std.testing.expect(approx(@floatCast(out.data[1]), 0.0290, 1e-3));
    try std.testing.expect(approx(@floatCast(out.data[2]), 0.0435, 1e-3));
}

test "ssm recurrence state persists across tokens" {
    const allocator = std.testing.allocator;
    var fixture = try buildDenseLayer(allocator);
    defer fixture.deinit();
    var layer = &fixture.layer;

    const t1 = [_]f32{ 1.0, 1.0, 1.0 };
    const t2 = [_]f32{ 0.3, -0.8, 1.1 };

    // Tras reset, el estado recurrente está a cero
    layer.resetState();
    for (layer.s_state) |v| try std.testing.expectEqual(@as(f32, 0.0), v);

    // Tras un token, el estado recurrente se ha escrito
    var x = try Tensor(f32).alloc(allocator, &.{ 1, test_dims.n_embd });
    defer x.deinit();
    for (0..3) |d| x.data[d] = @floatCast(t1[d]);
    var out = try Tensor(f32).alloc(allocator, &.{ 1, test_dims.n_embd });
    defer out.deinit();
    try layer.forward(x, &out, 1);
    var any_nonzero = false;
    for (layer.s_state) |v| {
        if (v != 0) any_nonzero = true;
    }
    try std.testing.expect(any_nonzero);

    // Determinismo: procesar el mismo token dos veces (con reset) es idéntico
    layer.resetState();
    var x2 = try Tensor(f32).alloc(allocator, &.{ 1, test_dims.n_embd });
    defer x2.deinit();
    for (0..3) |d| x2.data[d] = @floatCast(t2[d]);
    var out2 = try Tensor(f32).alloc(allocator, &.{ 1, test_dims.n_embd });
    defer out2.deinit();
    try layer.forward(x2, &out2, 1);
    layer.resetState();
    var out3 = try Tensor(f32).alloc(allocator, &.{ 1, test_dims.n_embd });
    defer out3.deinit();
    try layer.forward(x2, &out3, 1);
    try std.testing.expect(approx(@floatCast(out3.data[0]), @floatCast(out2.data[0]), 1e-6));
}

test "ssm forward matches brute-force reference" {
    const allocator = std.testing.allocator;
    var fixture = try buildTestLayer(allocator);
    defer fixture.deinit();
    var layer = &fixture.layer;

    const N: usize = 3;
    var x = try Tensor(f32).alloc(allocator, &.{ N, test_dims.n_embd });
    defer x.deinit();
    const x_vals = [_]f32{ 0.3, -0.8, 1.1, 2.0, 0.5, -1.3, -0.4, 0.9, 1.7 };
    for (0..N * test_dims.n_embd) |i| x.data[i] = @floatCast(x_vals[i]);

    var out = try Tensor(f32).alloc(allocator, &.{ N, test_dims.n_embd });
    defer out.deinit();
    try layer.forward(x, &out, N);

    // ─── Referencia independiente ───
    const p = layer.params;
    // Comptime: los tamaños vienen de test_dims (los de SsmParams son runtime)
    const qkv_dim = test_dims.d_state * test_dims.n_group * 2 + test_dims.d_inner;
    const key_dim = test_dims.d_state * test_dims.n_group;

    var z_saved: [N][test_dims.d_inner]f32 = undefined;
    var b_saved: [N][test_dims.dt_rank]f32 = undefined;
    var g_saved: [N][test_dims.dt_rank]f32 = undefined;
    var conv_saved: [N][qkv_dim]f32 = undefined;

    // Los pesos f16 (dequantizados) coinciden con los f32 originales porque
    // los valores del test son exactamente representables en f16.
    const qkv_w = std.mem.bytesAsSlice(f32, fixture.qkv_bytes);
    const z_w = std.mem.bytesAsSlice(f32, fixture.z_bytes);
    const out_w = std.mem.bytesAsSlice(f32, fixture.out_bytes);

    for (0..N) |t| {
        for (0..qkv_dim) |j| {
            var acc: f32 = 0;
            for (0..p.n_embd) |i| acc += @as(f32, @floatCast(x.data[t * p.n_embd + i])) * qkv_w[i + j * p.n_embd];
            const v = acc;
            conv_saved[t][j] = v / (1.0 + @exp(-v)); // conv kernel=1 + silu
        }
        for (0..p.d_inner) |j| {
            var acc: f32 = 0;
            for (0..p.n_embd) |i| acc += @as(f32, @floatCast(x.data[t * p.n_embd + i])) * z_w[i + j * p.n_embd];
            z_saved[t][j] = acc;
        }
        for (0..p.dt_rank) |j| {
            var acc: f32 = 0;
            for (0..p.n_embd) |i| acc += @as(f32, @floatCast(x.data[t * p.n_embd + i])) * layer.w_beta.data[j * p.n_embd + i];
            b_saved[t][j] = 1.0 / (1.0 + @exp(-acc));
        }
        for (0..p.dt_rank) |j| {
            var acc: f32 = 0;
            for (0..p.n_embd) |i| acc += @as(f32, @floatCast(x.data[t * p.n_embd + i])) * layer.w_alpha.data[j * p.n_embd + i];
            g_saved[t][j] = layer.ssm_a.data[j] * @log(1.0 + @exp(acc + layer.dt_bias.data[j]));
        }
    }

    // l2 norm q/k
    var qk_norm: [N][2][key_dim]f32 = undefined;
    for (0..N) |t| {
        for (0..2) |part| {
            const off = part * key_dim;
            for (0..p.n_group) |h| {
                var sum_sq: f32 = 0;
                for (0..p.d_state) |ii| {
                    const v = conv_saved[t][off + h * p.d_state + ii];
                    sum_sq += v * v;
                }
                // 1.14: ref = fórmula FLA (eps dentro de la raíz)
                const s = 1.0 / @sqrt(sum_sq + p.rms_eps);
                for (0..p.d_state) |ii| {
                    qk_norm[t][part][h * p.d_state + ii] = conv_saved[t][off + h * p.d_state + ii] * s;
                }
            }
        }
    }

    // recurrencia delta net
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(p.d_state)));
    var ref_state: [test_dims.d_state * test_dims.d_state * test_dims.dt_rank]f32 = [_]f32{0} ** (test_dims.d_state * test_dims.d_state * test_dims.dt_rank);
    var ref_attn: [N][test_dims.d_inner]f32 = undefined;
    for (0..N) |t| {
        for (0..p.dt_rank) |hv| {
            const hk = hv % p.n_group; // 7.1b: módulo, igual que la implementación
            const g = @exp(g_saved[t][hv]);
            const b = b_saved[t][hv];
            const s_base = hv * p.d_state * p.d_state;
            for (0..p.d_state * p.d_state) |i| ref_state[s_base + i] *= g;

            var d_buf: [4]f32 = undefined;
            for (0..p.d_state) |j| {
                var sk: f32 = 0;
                for (0..p.d_state) |i| {
                    sk += ref_state[s_base + i * p.d_state + j] * qk_norm[t][1][hk * p.d_state + i];
                }
                d_buf[j] = b * (conv_saved[t][2 * key_dim + hv * p.d_state + j] - sk);
            }
            for (0..p.d_state) |i| {
                const kv = qk_norm[t][1][hk * p.d_state + i];
                for (0..p.d_state) |j| {
                    ref_state[s_base + i * p.d_state + j] += kv * d_buf[j];
                }
            }
            for (0..p.d_state) |j| {
                var o: f32 = 0;
                for (0..p.d_state) |i| {
                    o += ref_state[s_base + i * p.d_state + j] * qk_norm[t][0][hk * p.d_state + i];
                }
                ref_attn[t][hv * p.d_state + j] = o * scale;
            }
        }
        // gated norm
        for (0..p.dt_rank) |hv| {
            const base = hv * p.d_state;
            var mean_sq: f32 = 0;
            for (0..p.d_state) |i| {
                const v = ref_attn[t][base + i];
                mean_sq += v * v;
            }
            mean_sq /= @as(f32, @floatFromInt(p.d_state));
            const rscale = 1.0 / @sqrt(mean_sq + p.rms_eps);
            for (0..p.d_state) |i| {
                const zn = z_saved[t][base + i];
                const silu = zn / (1.0 + @exp(-zn));
                ref_attn[t][base + i] = ref_attn[t][base + i] * rscale * layer.ssm_norm.data[i] * silu;
            }
        }
        // output projection
        var ref_out: [test_dims.n_embd]f32 = undefined;
        for (0..p.n_embd) |j| {
            var acc: f32 = 0;
            for (0..p.d_inner) |i| acc += ref_attn[t][i] * out_w[i + j * p.d_inner];
            ref_out[j] = acc;
        }
        for (0..p.n_embd) |d| {
            const got = out.data[t * p.n_embd + d];
            try std.testing.expect(approx(got, ref_out[d], 2e-2));
        }
    }
}

test "deltanet recurrence matches FLA naive reference" {
    // Valida decay (exp(-exp(A)*softplus)), pairing de k-heads (bloque
    // hv/(n_v/n_k)) y escala 1/sqrt(d_state) contra el kernel de FLA.
    // Dims: n_k_heads=2, n_v_heads=4 → ratio=2, ejercita hk != hv.
    const allocator = std.testing.allocator;
    const p = SsmParams{
        .n_embd = 3,
        .d_inner = 8, // value_dim = dt_rank * d_state
        .d_state = 2,
        .dt_rank = 4,
        .n_group = 2,
        .d_conv = 1,
    };
    var layer = try SsmLayer.init(allocator, 0, p, .auto);
    defer layer.deinit();
    layer.resetState();

    const key_dim = p.keyDim();
    const qkv_dim = p.qkvDim();
    const N: usize = 3;

    var conv_out = try Tensor(f32).alloc(allocator, &.{ N, qkv_dim });
    defer conv_out.deinit();
    for (0..N * qkv_dim) |i| {
        const t = i / qkv_dim;
        const c = i % qkv_dim;
        conv_out.data[i] = 0.1 * @as(f32, @floatFromInt(t + 1)) * @as(f32, @floatFromInt(c + 1)) +
            0.03 * @as(f32, @floatFromInt(c % 3));
    }
    var gate = try Tensor(f32).alloc(allocator, &.{ N, p.dt_rank });
    defer gate.deinit();
    for (0..N * p.dt_rank) |i| gate.data[i] = -0.4 - 0.1 * @as(f32, @floatFromInt(i % p.dt_rank));
    var beta = try Tensor(f32).alloc(allocator, &.{ N, p.dt_rank });
    defer beta.deinit();
    for (0..N * p.dt_rank) |i| beta.data[i] = 0.3 + 0.1 * @as(f32, @floatFromInt(i % p.dt_rank));
    var attn_out = try Tensor(f32).alloc(allocator, &.{ N, p.d_inner });
    defer attn_out.deinit();

    layer.deltaNetRecurrence(conv_out, gate, beta, attn_out, N, key_dim, p.n_group, p.dt_rank, p.d_state);

    // ─── referencia naive (fiel al kernel de FLA) ───
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(p.d_state)));
    var ref_S = [_][2][2]f32{
        .{ .{ 0, 0 }, .{ 0, 0 } },
        .{ .{ 0, 0 }, .{ 0, 0 } },
        .{ .{ 0, 0 }, .{ 0, 0 } },
        .{ .{ 0, 0 }, .{ 0, 0 } },
    };
    for (0..N) |t| {
        for (0..p.dt_rank) |hv| {
            const hk = hv % p.n_group; // 7.1b: módulo, igual que la implementación
            const g = @exp(gate.data[t * p.dt_rank + hv]);
            for (0..p.d_state) |i| {
                for (0..p.d_state) |j| ref_S[hv][i][j] *= g;
            }
            const v_base = t * qkv_dim + 2 * key_dim + hv * p.d_state;
            const k_base = t * qkv_dim + key_dim + hk * p.d_state;
            const q_base = t * qkv_dim + hk * p.d_state;
            var d: [2]f32 = undefined;
            for (0..p.d_state) |j| {
                var sk: f32 = 0;
                for (0..p.d_state) |i| sk += ref_S[hv][i][j] * conv_out.data[k_base + i];
                d[j] = beta.data[t * p.dt_rank + hv] * (conv_out.data[v_base + j] - sk);
            }
            for (0..p.d_state) |i| {
                for (0..p.d_state) |j| {
                    ref_S[hv][i][j] += conv_out.data[k_base + i] * d[j];
                }
            }
            for (0..p.d_state) |j| {
                var o: f32 = 0;
                for (0..p.d_state) |i| o += ref_S[hv][i][j] * conv_out.data[q_base + i];
                const want = o * scale;
                const got = attn_out.data[t * p.d_inner + hv * p.d_state + j];
                try std.testing.expectApproxEqAbs(want, got, 1e-5);
            }
        }
    }
    for (0..p.dt_rank) |hv| {
        for (0..p.d_state) |i| {
            for (0..p.d_state) |j| {
                try std.testing.expectApproxEqAbs(
                    ref_S[hv][i][j],
                    layer.s_state[hv * p.d_state * p.d_state + i * p.d_state + j],
                    1e-5,
                );
            }
        }
    }
}

test "1.14 GDN l2norm: fórmula FLA (eps dentro de la raíz, NO clamp)" {
    // Regresión 1.14: con vector casi-nulo (sum_sq < eps²) el clamp
    // `1/max(sqrt,eps)` y la fórmula FLA `1/sqrt(sum+eps)` divergen:
    //   FLA:   scale = 1/sqrt(2e-14+1e-6) ≈ 1000.5 → out ≈ 1e-4·√2
    //   clamp: scale = 1/1e-6 = 1e6        → out = 0.1·√2 (700× mayor)
    // FLA l2norm.py:69 `y = x/sqrt(sum(x²)+eps)` — eps DENTRO. OJO: FLA
    // NO fuerza norma-unitaria cuando sum ≪ eps (eps domina el denominador).
    const eps: f32 = 1e-6;
    const x = [_]f32{ 1e-7, 1e-7, -1e-7, 1e-7 };
    var sum_sq: f32 = 0;
    for (x) |v| sum_sq += v * v;
    const fla_scale = 1.0 / @sqrt(sum_sq + eps);
    const fla_out: [4]f32 = .{ x[0] * fla_scale, x[1] * fla_scale, x[2] * fla_scale, x[3] * fla_scale };
    // Norma FLA ≈ 1.4142e-4 (eps domina), NO 0.1414 (clamp):
    var fla_norm: f32 = 0;
    for (fla_out) |v| fla_norm += v * v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.4142136e-4), @sqrt(fla_norm), 1e-6);
    // El clamp habría dado norma ≈ sqrt(sum)/eps = 2e-5 (no-unitaria) —
    // este expect fallaría con la fórmula antigua.
    // Y l2NormQK usa exactamente esta fórmula (ssm.zig l2NormQK):
    const allocator = std.testing.allocator;
    const p = SsmParams{
        .n_embd = 3,
        .d_inner = 4,
        .d_state = 2,
        .dt_rank = 2,
        .n_group = 1,
        .d_conv = 1,
    };
    var layer = try SsmLayer.init(allocator, 0, p, .auto);
    defer layer.deinit();
    var conv_out = try Tensor(f32).alloc(allocator, &.{ 1, p.qkvDim() });
    defer conv_out.deinit();
    // q head (offset 0) casi-nula; el resto (k,v) valores normales.
    const qkv_dim = p.qkvDim();
    const key_dim = p.keyDim();
    for (0..qkv_dim) |i| conv_out.data[i] = 0.5;
    for (0..p.d_state) |i| conv_out.data[i] = x[i % 4];
    layer.l2NormQK(conv_out, 1, key_dim, 1, p.d_state);
    // q normalizado con fórmula FLA: norma ≈ 1.4142e-4 (eps domina el
    // denominador), NO 0.1414 (clamp — fallaría este expect):
    var out_norm: f32 = 0;
    for (0..p.d_state) |i| out_norm += conv_out.data[i] * conv_out.data[i];
    try std.testing.expectApproxEqAbs(@as(f32, 1.4142136e-4), @sqrt(out_norm), 1e-6);
}
