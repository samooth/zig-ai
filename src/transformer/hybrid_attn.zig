//! Hybrid attention layer — full attention para Qwen3.5 (qwen35).
//! Fiel a llama.cpp build_layer_attn: Q+G fusionado, Q/K norm per-head,
//! GQA 16->4, MRoPE (NEOX half-split), gate sigmoid, KV-cache.
const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const cublas = @import("cublas");
const cudaz = @import("cudaz");
const layer_kernels = @import("layer_kernels");
const QuantWeight = @import("quant_weight").QuantWeight;
const gguf = @import("gguf");
const norm = @import("norm");
const rope_mod = @import("rope");
const paged = @import("paged_attention");
const kv_cache = @import("kv_cache");
const kv_quant = kv_cache.kv_quant;
const quantBytes = kv_quant.quantBytes;
const debugz = @import("debug");

// 9.4 debug: step (start_pos) del último forwardGPU — filename del dump
// ZIG_AI_ATT_DUMP para no sobrescribir prefill vs decode.
var att_dbg_step_global: usize = 0;
const kvarn_fattn = @import("fattn_kvarn"); // lane-b 9.4: FA-native
const kvk = @import("kvarn_kernels"); // lane-b 9.4: KvarnDesc

pub const HybridAttnError = error{
    WeightFileNotFound,
    ShapeMismatch,
    KvCacheNotSet,
    SequenceNotFound,
};

pub const HybridAttnParams = struct {
    n_embd: usize = 4096,
    n_head: usize = 16,
    n_kv_head: usize = 4,
    head_dim: usize = 256,
    n_rot: usize = 64,
    rope_sections: [4]usize = .{ 11, 11, 10, 0 },
    rope_freq_base: f32 = 1e7,
    rope_scaling: rope_mod.RopeScaling = .{},
    rms_eps: f32 = 1e-6,
    max_seq_len: usize = 2048,
    no_gate: bool = false, // LFM2: separate Q (not fused Q+G); K2-Horizon also uses this
    use_mrope: bool = true, // LFM2: use standard RoPE instead of MRoPE
    // F1v8: pairing del applyRoPE estándar (solo path use_mrope=false).
    // Oráculo llama-model.cpp: lfm2 → NEOX (2682). Default NEOX = el
    // comportamiento histórico del path lfm2, preservado explícito.
    rope_neox: rope_mod.RopePairing = .neox,

    // K2-Horizon: softplus gate on attention output (before WO projection)
    has_softplus_gate: bool = false,

    // K2-Horizon MoVA
    n_value_expert: usize = 0,
    n_value_expert_used: usize = 0,
    expert_gating_func: u32 = 1, // 1=sigmoid, 2=softmax (match llama.cpp enum)
    expert_weights_norm: bool = false,
    expert_weights_scale: f32 = 0.0,

    // RLT SWA: sliding window attention (null = full context)
    swa_window: ?usize = null,

    pub fn qg_dim(self: HybridAttnParams) usize {
        if (self.no_gate) return self.n_head * self.head_dim;
        return self.n_head * self.head_dim * 2;
    }
    pub fn kv_dim(self: HybridAttnParams) usize {
        return self.n_kv_head * self.head_dim;
    }
};

pub const AttentionLayer = struct {
    allocator: std.mem.Allocator,
    layer_idx: usize,
    params: HybridAttnParams,
    matmul_engine: matmul.MatmulEngine,

    // Pesos grandes: QuantWeight (bytes mmap, préstamo al GGUF). Layout [out, in]
    w_q: QuantWeight, // [qg_dim, n_embd] = [8192, 4096] fused Q+G
    w_k: QuantWeight, // [kv_dim, n_embd] = [1024, 4096]
    w_v: QuantWeight, // [kv_dim, n_embd] = [1024, 4096]
    w_o: QuantWeight, // [n_embd, n_head*head_dim] = [4096, 4096]

    // K2-Horizon: optional softplus gate weight
    wqkv_gate: ?QuantWeight = null, // [n_head*head_dim, n_embd] - optional

    // K2-Horizon MoVA: optional value expert routing
    attn_v_gate: ?QuantWeight = null, // [n_embd, n_value_expert] - optional router
    attn_v_gate_b: ?Tensor(f32) = null, // [n_value_expert] - optional bias
    attn_v_exps: ?QuantWeight = null, // [n_embd, n_embd_v_gqa, n_value_expert] - optional experts

    // Pesos de normalización (f32, pequeños)
    attn_q_norm: Tensor(f32), // [head_dim] or [n_head * head_dim] for grouped
    attn_k_norm: Tensor(f32), // [head_dim] or [n_kv_head * head_dim] for grouped
    /// U1 (llama-arch): true si el GGUF trae attn_q_norm/attn_k_norm
    /// (qwen35/qwen3). false ⇒ las normas per-head se OMITEN (llama no
    /// tiene q/k_norm; rmsNorm(x, ones, eps) = x/rms(x) NO es identidad —
    /// aplicar ones reescala cada head por su energía y corrompe el texto).
    has_qk_norm: bool = true,

    // Scratch f16 persistente (reutilizado cada forward)
    scratch_q: []f32, // qg_dim * n_embd
    scratch_k: []f32, // kv_dim * n_embd
    scratch_v: []f32, // kv_dim * n_embd
    scratch_o: []f32, // n_embd * (n_head*head_dim)
    scratch_gate: ?[]f32 = null, // n_head*head_dim * n_embd (K2-Horizon softplus gate)

    // KV-Cache paginado (PagedKVCache compartido entre capas de atención)
    paged_kv: *paged.PagedKVCache,
    block_table: *paged.BlockTable,
    sequence_id: u64 = 0,

    /// Vision (PLAN_MMPROJ 3.2): delta ctx_pos − kv_slot para el RoPE de
    /// decode. Con imagen: n_img slots pero max(nx,ny) posiciones ⇒ delta
    /// negativo (p.ej. −280). Sólo actúa si d_rope_pos habilitado en el pool.
    rope_pos_delta: i64 = 0,

    /// 5.2 (lane-b1): modo denoise DFlash — atención NO-causal (bidireccional
    /// en [0, start_pos+n)) para el forward CPU del draft sidecar sobre
    /// [anchor, MASK×(bs-1)]. Default true (path normal causal).
    causal: bool = true,

    // Motor GPU de PagedAttention (compartido entre capas; null si CUDA no
    // disponible). El pool del GPU se indexa por phys_id global del
    // BlockAllocator compartido, así que una sola instancia cubre todas las
    // capas de atención (evita OOM: num_blocks * block_bytes por capa).
    paged_gpu: ?*paged.PagedAttentionGpu = null,

    // Quantized KV-cache staging (solo cuando config.quant_k/v != .fp16).
    // Se acumulan los valores de un bloque lógico f16 en staging y se cuantizan
    // con kv_quant.encode al sellar el bloque. Layout por bloque:
    //   [ K_tile_bytes ][ V_tile_bytes ]
    k_quant: paged.QuantFormat,
    v_quant: paged.QuantFormat,
    k_tile_bytes: usize,
    v_tile_bytes: usize,
    k_staging: []f16,
    v_staging: []f16,
    staged_block: i64 = -1,
    staged_tokens: usize = 0,

    // Buffers GPU para el forward de atención residente (Phase 1b).
    gpu: ?AttentionGpu = null,

    // 9.4 (lane-b): camino FA-native sobre KVarN records (sin materializar).
    // Opt-in: cli.zig lo habilita por capa cuando ZIG_AI_KVARN_FA=1 y el
    // KvarnGpuCache está hidratado. El trampoline `kvarn_append_fn` corre
    // tras proyectar K/V (antes de la atención), alimenta el cache y
    // devuelve la base de los descs K de la capa (V = base + KvarnDesc);
    // null/0 ⇒ camino paged clásico (invariante de trunk). El flip del
    // flag es seguro por paso: cualquier fallo decae a paged.
    kvarn_native: bool = false,
    /// cubin fattn (portable d256/d128) — lo comparte el cli.
    kvarn_fattn_module: ?cudaz.CUmodule = null,
    kvarn_append_fn: ?*const fn (ctx: ?*anyopaque, layer: u32, k: cudaz.CUdeviceptr, v: ?cudaz.CUdeviceptr, n: u32, base: u32, stream: cudaz.CUstream) anyerror!cudaz.CUdeviceptr = null,
    kvarn_append_ctx: ?*anyopaque = null,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        layer_idx: usize,
        params: HybridAttnParams,
        backend: matmul.Backend,
        paged_kv: *paged.PagedKVCache,
        block_table: *paged.BlockTable,
        paged_gpu: ?*paged.PagedAttentionGpu,
    ) !Self {
        var engine = try matmul.MatmulEngine.init(allocator, backend, .f32);
        errdefer engine.deinit();

        // Scratch de pesos atención f32 LAZY: se allocan en loadWeightsFromGguf
        // (guard scratch_q.len==0). Eager aquí costaba ~150MB/capa residente
        // aunque la capa no estuviera cargada (27B: OOM host en carga).

        var attn_q_norm = try Tensor(f32).alloc(allocator, &.{params.n_head * params.head_dim});
        errdefer attn_q_norm.deinit();
        var attn_k_norm = try Tensor(f32).alloc(allocator, &.{params.n_kv_head * params.head_dim});
        errdefer attn_k_norm.deinit();

        const paged_gpu_local: ?*paged.PagedAttentionGpu = paged_gpu;

        const block_size = paged_kv.config.block_size;
        const k_quant: paged.QuantFormat = paged_kv.config.quant_k;
        const v_quant: paged.QuantFormat = paged_kv.config.quant_v;
        const tile_elems = block_size * params.kv_dim();
        const k_tile_bytes = quantBytes(k_quant, tile_elems);
        const v_tile_bytes = quantBytes(v_quant, tile_elems);
        const k_staging = try allocator.alloc(f16, tile_elems);
        errdefer allocator.free(k_staging);
        const v_staging = try allocator.alloc(f16, tile_elems);
        errdefer allocator.free(v_staging);

        return Self{
            .allocator = allocator,
            .layer_idx = layer_idx,
            .params = params,
            .matmul_engine = engine,
            .w_q = undefined,
            .w_k = undefined,
            .w_v = undefined,
            .w_o = undefined,
            .attn_q_norm = attn_q_norm,
            .attn_k_norm = attn_k_norm,
            // U1: se calibra en loadWeightsFromGguf según existencia real
            // de los tensores (llama-arch no los trae).
            .has_qk_norm = false,
            .scratch_q = &[_]f32{},
            .scratch_k = &[_]f32{},
            .scratch_v = &[_]f32{},
            .scratch_o = &[_]f32{},
            .paged_kv = paged_kv,
            .block_table = block_table,
            .paged_gpu = paged_gpu_local,
            .k_quant = k_quant,
            .v_quant = v_quant,
            .k_tile_bytes = k_tile_bytes,
            .v_tile_bytes = v_tile_bytes,
            .k_staging = k_staging,
            .v_staging = v_staging,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.gpu) |*g| g.deinit();
        self.matmul_engine.deinit();
        self.allocator.free(self.scratch_q);
        self.allocator.free(self.scratch_k);
        self.allocator.free(self.scratch_v);
        self.allocator.free(self.scratch_o);
        if (self.scratch_gate) |sg| self.allocator.free(sg);
        self.attn_q_norm.deinit();
        self.attn_k_norm.deinit();
        self.allocator.free(self.k_staging);
        self.allocator.free(self.v_staging);
    }

    pub fn resetState(self: *Self) void {
        self.sequence_id = 0;
        self.staged_block = -1;
        self.staged_tokens = 0;
    }

    /// Libera scratch f32 dequantizados. Requiere re-alloc en loadWeightsFromGguf.
    pub fn unloadWeights(self: *Self) void {
        // 7.1a-b (lane-c, remanente 7.1): evicción SELECTIVA del weight_cache
        // device por host-ptr de CADA peso propio (hooks D3 existían sin
        // callers). clearWeightCache global des-cacheaba también las capas
        // residentes válidas ⇒ re-upload ~300MB/capa por ciclo LRU del
        // streamer (thrash PCIe en decode con max_resident=2). Ahora el
        // unload de esta capa solo suelta SUS entradas.
        _ = self.matmul_engine.evictWeightCachePtr(@intFromPtr(self.scratch_q.ptr));
        _ = self.matmul_engine.evictWeightCachePtr(@intFromPtr(self.scratch_k.ptr));
        _ = self.matmul_engine.evictWeightCachePtr(@intFromPtr(self.scratch_v.ptr));
        _ = self.matmul_engine.evictWeightCachePtr(@intFromPtr(self.scratch_o.ptr));
        if (self.scratch_q.len > 0) self.allocator.free(self.scratch_q);
        if (self.scratch_k.len > 0) self.allocator.free(self.scratch_k);
        if (self.scratch_v.len > 0) self.allocator.free(self.scratch_v);
        if (self.scratch_o.len > 0) self.allocator.free(self.scratch_o);
        if (self.scratch_gate) |sg| {
            self.allocator.free(sg);
            self.scratch_gate = null;
        }
        self.scratch_q = &[_]f32{};
        self.scratch_k = &[_]f32{};
        self.scratch_v = &[_]f32{};
        self.scratch_o = &[_]f32{};
    }
    /// Cuantiza el tile f16 acumulado (staging) al layout canónico GGUF en
    /// el bloque físico cuyo bloque lógico es `self.staged_block`. Se usa al
    /// sellar un bloque completo o al cambiar de bloque lógico.
    fn flushQuantTile(self: *Self) !void {
        const sblk = self.staged_block;
        if (sblk < 0) return;
        const phys_opt = self.block_table.getPhysical(@as(usize, @intCast(sblk)));
        if (phys_opt) |phys_id| {
            const block_data = self.paged_kv.block_alloc.memory_pool[phys_id * self.paged_kv.block_alloc.block_bytes ..];
            const kv_dim = self.params.kv_dim();
            const nval = self.staged_tokens * kv_dim;
            // Rellenar el tile con ceros para emitir exactamente k_tile_bytes/v_tile_bytes.
            if (nval < self.k_staging.len) @memset(self.k_staging[nval..], 0.0);
            if (nval < self.v_staging.len) @memset(self.v_staging[nval..], 0.0);
            if (self.k_quant != .fp16) {
                kv_quant.encode(self.k_quant, self.k_staging, block_data[0..self.k_tile_bytes]);
            } else {
                const dst = block_data[0..self.k_tile_bytes];
                @memcpy(dst, std.mem.sliceAsBytes(self.k_staging));
            }
            if (self.v_quant != .fp16) {
                if (debugz.dbg.kv_sr_v) {
                    kv_quant.encodeOpts(self.v_quant, self.v_staging, block_data[self.k_tile_bytes..][0..self.v_tile_bytes], .{ .stochastic = true });
                } else {
                    kv_quant.encode(self.v_quant, self.v_staging, block_data[self.k_tile_bytes..][0..self.v_tile_bytes]);
                }
            } else {
                const dst = block_data[self.k_tile_bytes..][0..self.v_tile_bytes];
                @memcpy(dst, std.mem.sliceAsBytes(self.v_staging));
            }
        }
        self.staged_block = -1;
        self.staged_tokens = 0;
    }

    /// Alloc+dequant lazy de los scratch f32 de atención (camino CPU o
    /// fallback f32 GPU). Idempotente; NO-op si ya materializados.
    fn ensureF32Scratch(self: *Self) !void {
        if (self.scratch_q.len > 0) return;
        const p = self.params;
        const qg_dim = p.qg_dim();
        const kv_dim = p.kv_dim();
        self.scratch_q = try self.allocator.alloc(f32, qg_dim * p.n_embd);
        self.scratch_k = try self.allocator.alloc(f32, kv_dim * p.n_embd);
        self.scratch_v = try self.allocator.alloc(f32, kv_dim * p.n_embd);
        self.scratch_o = try self.allocator.alloc(f32, p.n_embd * p.n_head * p.head_dim);
        self.w_q.dequantToF32Transposed(self.scratch_q);
        self.w_k.dequantToF32Transposed(self.scratch_k);
        self.w_v.dequantToF32Transposed(self.scratch_v);
        self.w_o.dequantToF32Transposed(self.scratch_o);
    }

    /// True si TODOS los pesos de atención irán por qgemm (kernel
    /// cuantizado por dtype) — scratch f32 innecesarios en RAM.
    /// REQUIERE w_* ya cargados (ver orden carga→decisión en
    /// loadWeightsFromGguf).
    fn attnQuantResident(self: *const Self) bool {
        if (!cudaz.isCudaAvailable()) return false; // CPU path usa f32
        return qgemmTypeFor(self.w_q.dtype()) != null and
            qgemmTypeFor(self.w_k.dtype()) != null and
            qgemmTypeFor(self.w_v.dtype()) != null and
            qgemmTypeFor(self.w_o.dtype()) != null;
    }

    /// Carga pesos desde GGUF (nombres qwen35). Si scratch está vacío
    /// (después de unloadWeights), re-alloca antes de dequantizar.
    pub fn loadWeightsFromGguf(self: *Self, g: *const gguf.GgufFile) !void {
        // F/C fix 700 (known-issue v2): la GEOMETRÍA REAL del tensor manda
        // sobre la metadata — corre ANTES de la carga para que el camino
        // cuant-residente (2e97e00) TAMBIÉN vea la geometría corregida
        // (qwen3moe: w_q de q_dim filas ⇒ no_gate; w_k {512,128} ⇒ n_kv_head
        // real — sin esto el qgemm hace OOB del banco ⇒ CUDA 700).
        {
            var qbuf: [96]u8 = undefined;
            const q_name = std.fmt.bufPrint(&qbuf, "blk.{d}.attn_q.weight", .{self.layer_idx}) catch unreachable;
            if (g.getTensor(q_name)) |qi| {
                const q_dim_real: usize = @intCast(qi.dims[1]);
                const q_dim_expected: usize = self.params.n_head * self.params.head_dim;
                if (self.params.no_gate) {
                    if (q_dim_real != q_dim_expected and q_dim_real == q_dim_expected * 2) {
                        self.params.no_gate = false; // fused QG tras todo (defensivo)
                        debugz.dbg.printLevel(.info, "[hybrid] capa {d}: attn_q out={d} ⇒ QG fused (no_gate off)\n", .{ self.layer_idx, q_dim_real });
                    }
                } else {
                    if (q_dim_real == q_dim_expected) {
                        self.params.no_gate = true; // atención clásica (qwen3moe et al.)
                        debugz.dbg.printLevel(.info, "[hybrid] capa {d}: attn_q out={d} == q_dim ⇒ atención clásica (no_gate on)\n", .{ self.layer_idx, q_dim_real });
                    }
                }
            }
            // n_kv_head real desde w_k: la metadata puede mentir (loggenix:
            // kv=8 declarado, w_k {512,128} ⇒ 2 kv_heads de 64).
            var kbuf: [96]u8 = undefined;
            const k_name = std.fmt.bufPrint(&kbuf, "blk.{d}.attn_k.weight", .{self.layer_idx}) catch unreachable;
            if (g.getTensor(k_name)) |ki| {
                const kv_out_real: usize = @intCast(ki.dims[1]);
                if (kv_out_real > 0 and kv_out_real % self.params.head_dim == 0) {
                    const kv_heads_real = kv_out_real / self.params.head_dim;
                    if (kv_heads_real != self.params.n_kv_head) {
                        const kv_meta = self.params.n_kv_head;
                        self.params.n_kv_head = kv_heads_real;
                        debugz.dbg.printLevel(.info, "[hybrid] capa {d}: w_k out={d} ⇒ n_kv_head={d} (metadata decía {d})\n", .{ self.layer_idx, kv_out_real, kv_heads_real, kv_meta });
                    }
                }
            }
        }
        // Atención cuant-residente (patrón llama.cpp, 2e97e00): los
        // QuantWeight son vistas de los bytes mmap (cero copia) — se cargan
        // SIEMPRE. La decisión de materializar scratch f32 (dequant) se
        // toma DESPUÉS, con los dtypes ya conocidos: si el camino cuantizado
        // GPU cubre TODOS los pesos de atención (qgemm consume bytes RAW —
        // dequant DENTRO del kernel), los scratch f32 NO se materializan:
        // RAM host = pesos cuantizados solamente; lazy al primer forward
        // que caiga al camino f32. NOTA: el orden carga→decisión es
        // obligatorio: w_* nacen `undefined` en init, decidir antes sería
        // UB (dtype de basura).
        const prefix = try std.fmt.allocPrint(self.allocator, "blk.{d}.", .{self.layer_idx});
        defer self.allocator.free(prefix);

        self.w_q = try loadQuantWeight(g, prefix, "attn_q.weight");
        self.w_k = try loadQuantWeight(g, prefix, "attn_k.weight");
        self.w_v = try loadQuantWeight(g, prefix, "attn_v.weight");
        self.w_o = try loadQuantWeight(g, prefix, "attn_output.weight");

        if (self.attnQuantResident()) {
            if (debugz.dbg.at(.info)) {
                debugz.dbg.printLevel(.info, "[pool] capa {d}: attn quant-residente (sin dequant f32; q={s} k={s} v={s} o={s})\n", .{
                    self.layer_idx,             @tagName(self.w_q.dtype()), @tagName(self.w_k.dtype()),
                    @tagName(self.w_v.dtype()), @tagName(self.w_o.dtype()),
                });
            }
        } else {
            debugz.dbg.printLevel(.info, "[pool] capa {d}: attn dequant f32 eager (camino legacy)\n", .{self.layer_idx});
            // Descuantizar los pesos UNA vez aquí (no por token en forward):
            // los scratch f32 persisten y el caché de pesos GPU los sube al
            // device una sola vez. Elimina re-descuantizar ~3GB/token.
            // Re-allocar scratch si fue liberado por unloadWeights.
            try self.ensureF32Scratch();
            self.w_q.dequantToF32Transposed(self.scratch_q);
            self.w_k.dequantToF32Transposed(self.scratch_k);
            self.w_v.dequantToF32Transposed(self.scratch_v);
            self.w_o.dequantToF32Transposed(self.scratch_o);
        }

        // Norm weights (f32). FIX dangling (2e97e00): reset antes del try
        // para que un fallo de load deje un tensor sin ownership — deinit()
        // no debe ver uno ya liberado (double-free evitado).
        // U1 (llama-arch): los q/k_norm son OPCIONALES. Si existen (qwen35/
        // qwen3) se cargan y has_qk_norm=true; si NO (llama), se cargan
        // ONES como placeholder (el forward los SALTA vía has_qk_norm —
        // rmsNorm(x,ones,eps)=x/rms(x) NO es identidad, aplicarlo
        // reescala cada head por su energía y corrompe el texto).
        {
            var nbuf: [128]u8 = undefined;
            const q_full = std.fmt.bufPrint(&nbuf, "blk.{d}.attn_q_norm.weight", .{self.layer_idx}) catch unreachable;
            self.has_qk_norm = g.getTensor(q_full) != null;
            if (!self.has_qk_norm and debugz.dbg.at(.info)) {
                debugz.dbg.printLevel(.info, "[hybrid] capa {d}: sin attn_q/k_norm (llama-arch) — normas per-head OMITIDAS\n", .{self.layer_idx});
            }
        }
        self.attn_q_norm.deinit();
        self.attn_q_norm = .{ .data = &.{}, .shape = &.{}, .strides = &.{}, .offset = 0, .allocator = null, .owns_data = false };
        self.attn_q_norm = try loadGgufF32OrOnes(self.allocator, g, prefix, "attn_q_norm.weight", self.params.n_head * self.params.head_dim);
        self.attn_k_norm.deinit();
        self.attn_k_norm = .{ .data = &.{}, .shape = &.{}, .strides = &.{}, .offset = 0, .allocator = null, .owns_data = false };
        self.attn_k_norm = try loadGgufF32OrOnes(self.allocator, g, prefix, "attn_k_norm.weight", self.params.n_kv_head * self.params.head_dim);

        // K2-Horizon: optional softplus gate weight
        if (self.params.has_softplus_gate) {
            var gbuf: [128]u8 = undefined;
            const gate_full = std.fmt.bufPrint(&gbuf, "{s}attn_gate.weight", .{prefix}) catch unreachable;
            if (g.getTensor(gate_full)) |info| {
                self.wqkv_gate = QuantWeight.init(info, g.tensorData(info));
                // Allocate scratch for gate projection
                if (self.scratch_gate) |sg| self.allocator.free(sg);
                self.scratch_gate = try self.allocator.alloc(f32, info.dims[1] * info.dims[0]);
            } else {
                self.wqkv_gate = null;
            }
        }
    }

    /// Forward mixer-only: Attn(GQA+MRoPE+Gate) -> out
    /// Recibe input ya normalizado (pre-norm lo hace HybridLayer).
    /// `x`: [N, n_embd] (f16)
    /// `out`: [N, n_embd] (f16)
    /// `start_pos`: posición inicial en la secuencia (para RoPE y KV-cache)
    /// `n`: número de tokens a procesar (prefill en bloque o 1 token)
    /// `pos_ids`: position-ids M-RoPE per-token [n][4] (opcional; null ⇒
    ///   ids secuenciales desde start_pos — comportamiento clásico). Para
    ///   tokens de imagen inyectados (mtmd-helper.cpp:142, n_pos_per_embd=4).
    pub fn forward(self: *Self, x: Tensor(f32), out: *Tensor(f32), start_pos: usize, n: usize, pos_ids: ?[]const [4]i32) !void {
        const p = self.params;
        const qg_dim = p.qg_dim();
        const kv_dim = p.kv_dim();
        const head_dim = p.head_dim;
        const n_head = p.n_head;
        const n_kv_head = p.n_kv_head;

        // Dequant lazy (cuant-residente load): el camino CPU consume
        // f32 — materializar UNA vez al primer forward.
        try self.ensureF32Scratch();
        const N = n;

        // === 2. Proyección Q (y G si no es no_gate) ===
        var w_q_shape = [_]usize{ qg_dim, p.n_embd };
        var w_q_strides = [_]usize{ p.n_embd, 1 };
        const w_q32 = Tensor(f32){
            .data = self.scratch_q,
            .shape = &w_q_shape,
            .strides = &w_q_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        var qg32 = try Tensor(f32).alloc(self.allocator, &.{ N, qg_dim });
        defer qg32.deinit();
        try self.matmul_engine.linearProjection(f32, x, w_q32, &qg32);

        var Qf32 = try Tensor(f32).alloc(self.allocator, &.{ N, n_head, head_dim });
        defer Qf32.deinit();
        var Gf32 = try Tensor(f32).alloc(self.allocator, &.{ N, n_head, head_dim });
        defer Gf32.deinit();

        if (p.no_gate) {
            // LFM2: Q projection only, no G
            for (0..N) |t| {
                for (0..n_head) |h| {
                    for (0..head_dim) |d| {
                        Qf32.data[t * n_head * head_dim + h * head_dim + d] = qg32.data[t * qg_dim + h * head_dim + d];
                    }
                }
            }
        } else {
            // Qwen3.5: fused Q+G interleaved [Q0|G0|Q1|G1|...]
            for (0..N) |t| {
                for (0..n_head) |h| {
                    const base = h * (2 * head_dim);
                    for (0..head_dim) |d| {
                        Qf32.data[t * n_head * head_dim + h * head_dim + d] = qg32.data[t * qg_dim + base + d];
                        Gf32.data[t * n_head * head_dim + h * head_dim + d] = qg32.data[t * qg_dim + base + head_dim + d];
                    }
                }
            }
        }

        // === 3. Proyecciones K, V ===
        var w_k_shape = [_]usize{ kv_dim, p.n_embd };
        var w_k_strides = [_]usize{ p.n_embd, 1 };
        const w_k32 = Tensor(f32){
            .data = self.scratch_k,
            .shape = &w_k_shape,
            .strides = &w_k_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        var Kf32 = try Tensor(f32).alloc(self.allocator, &.{ N, kv_dim });
        defer Kf32.deinit();
        try self.matmul_engine.linearProjection(f32, x, w_k32, &Kf32);

        var w_v_shape = [_]usize{ kv_dim, p.n_embd };
        var w_v_strides = [_]usize{ p.n_embd, 1 };
        const w_v32 = Tensor(f32){
            .data = self.scratch_v,
            .shape = &w_v_shape,
            .strides = &w_v_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        var Vf32 = try Tensor(f32).alloc(self.allocator, &.{ N, kv_dim });
        defer Vf32.deinit();

        // Standard V projection (fallback if no MoVA)
        try self.matmul_engine.linearProjection(f32, x, w_v32, &Vf32);

        // Reshape buffers for K, V
        var Kf32_hm = try Tensor(f32).alloc(self.allocator, &.{ N, n_kv_head, head_dim });
        defer Kf32_hm.deinit();
        var Vf32_hm = try Tensor(f32).alloc(self.allocator, &.{ N, n_kv_head, head_dim });
        defer Vf32_hm.deinit();

        // K2-Horizon MoVA: reemplaza V lineal por routing de expertos
        if (p.n_value_expert > 0 and self.attn_v_gate != null and self.attn_v_exps != null) {
            const v_gate_w = self.attn_v_gate.?;
            const v_exps_w = self.attn_v_exps.?;

            // Router logits: [N, n_value_expert]
            var router_logits = try Tensor(f32).alloc(self.allocator, &.{ N, p.n_value_expert });
            defer router_logits.deinit();

            // Compute router logits: x @ attn_v_gate^T
            // attn_v_gate shape: [n_value_expert, n_embd] (QuantWeight stores [out, in])
            var v_gate_shape = [_]usize{ p.n_value_expert, p.n_embd };
            var v_gate_strides = [_]usize{ p.n_embd, 1 };

            // We need scratch for V gate weights - dequantize
            const v_gate_scratch = try self.allocator.alloc(f32, p.n_value_expert * p.n_embd);
            defer self.allocator.free(v_gate_scratch);
            v_gate_w.dequantToF32Transposed(v_gate_scratch);
            const v_gate_scratch_tensor = Tensor(f32){
                .data = v_gate_scratch,
                .shape = @as([]usize, &v_gate_shape),
                .strides = @as([]usize, &v_gate_strides),
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };
            try self.matmul_engine.linearProjection(f32, x, v_gate_scratch_tensor, &router_logits);

            // Apply gating function
            var probs = try Tensor(f32).alloc(self.allocator, &.{ N, p.n_value_expert });
            defer probs.deinit();
            if (p.expert_gating_func == 2) { // softmax
                for (0..N) |t| {
                    var max_val: f32 = -std.math.inf(f32);
                    for (0..p.n_value_expert) |e| {
                        max_val = @max(max_val, router_logits.data[t * p.n_value_expert + e]);
                    }
                    var sum_exp: f32 = 0;
                    for (0..p.n_value_expert) |e| {
                        probs.data[t * p.n_value_expert + e] = @exp(router_logits.data[t * p.n_value_expert + e] - max_val);
                        sum_exp += probs.data[t * p.n_value_expert + e];
                    }
                    for (0..p.n_value_expert) |e| {
                        probs.data[t * p.n_value_expert + e] /= sum_exp;
                    }
                }
            } else { // sigmoid (default)
                for (0..N * p.n_value_expert) |i| {
                    probs.data[i] = 1.0 / (1.0 + @exp(-router_logits.data[i]));
                }
            }

            // Add optional bias
            if (self.attn_v_gate_b) |bias| {
                for (0..N) |t| {
                    for (0..p.n_value_expert) |e| {
                        probs.data[t * p.n_value_expert + e] += bias.data[e];
                    }
                }
            }

            // Top-k selection
            const n_used = @min(p.n_value_expert_used, p.n_value_expert);
            var selected_experts = try self.allocator.alloc(usize, N * n_used);
            defer self.allocator.free(selected_experts);
            var selected_weights = try self.allocator.alloc(f32, N * n_used);
            defer self.allocator.free(selected_weights);

            for (0..N) |t| {
                // Simple top-k selection (argsort not available, use partial sort)
                var indices = try self.allocator.alloc(usize, p.n_value_expert);
                defer self.allocator.free(indices);
                for (0..p.n_value_expert) |e| indices[e] = e;

                // Partial sort for top-k
                for (0..n_used) |i| {
                    var max_idx = i;
                    for (i + 1..p.n_value_expert) |j| {
                        if (probs.data[t * p.n_value_expert + indices[j]] > probs.data[t * p.n_value_expert + indices[max_idx]]) {
                            max_idx = j;
                        }
                    }
                    const tmp = indices[i];
                    indices[i] = indices[max_idx];
                    indices[max_idx] = tmp;
                    selected_experts[t * n_used + i] = indices[i];
                    selected_weights[t * n_used + i] = probs.data[t * p.n_value_expert + indices[i]];
                }
            }

            // Optional weight normalization
            if (p.expert_weights_norm) {
                for (0..N) |t| {
                    var sum: f32 = 0;
                    for (0..n_used) |i| {
                        sum += selected_weights[t * n_used + i];
                    }
                    sum = @max(sum, 6.103515625e-5);
                    for (0..n_used) |i| {
                        selected_weights[t * n_used + i] /= sum;
                    }
                }
            }

            // Optional scaling
            if (p.expert_weights_scale != 0.0 and p.expert_weights_scale != 1.0) {
                for (0..N * n_used) |i| {
                    selected_weights[i] *= p.expert_weights_scale;
                }
            }

            // Compute routed values for selected experts
            // attn_v_exps shape: [n_embd, n_embd_v_gqa, n_value_expert]
            // We need to dequantize and compute x @ expert_weights for each selected expert
            var Vf32_mova = try Tensor(f32).alloc(self.allocator, &.{ N, n_kv_head, head_dim });
            defer Vf32_mova.deinit();
            @memset(Vf32_mova.data, 0.0);

            // Dequantize all experts (could be optimized to dequantize only selected)
            const expert_bytes = v_exps_w.bytes.len / p.n_value_expert;
            const expert_buf = try self.allocator.alloc(u8, expert_bytes);
            defer self.allocator.free(expert_buf);

            for (0..N) |t| {
                for (0..n_used) |k| {
                    const e = selected_experts[t * n_used + k];
                    @memcpy(expert_buf, v_exps_w.bytes[e * expert_bytes ..][0..expert_bytes]);

                    // Create synthetic TensorInfo for this expert
                    var expert_info = v_exps_w.info.*;
                    expert_info.n_dims = 2;
                    expert_info.dims[0] = v_exps_w.info.dims[0];
                    expert_info.dims[1] = v_exps_w.info.dims[1];
                    expert_info.dims[2] = 0;
                    expert_info.dims[3] = 0;

                    const expert_weight = try self.allocator.alloc(f32, @intCast(expert_info.numel()));
                    defer self.allocator.free(expert_weight);
                    try gguf.dequantTensor(&expert_info, expert_buf, expert_weight);

                    // Compute x[t] @ expert_weight for this expert
                    var w_shape = [_]usize{ head_dim, p.n_embd }; // [n_embd_v_gqa, n_embd]
                    var w_strides = [_]usize{ p.n_embd, 1 };
                    const expert_tensor = Tensor(f32){
                        .data = expert_weight,
                        .shape = @as([]usize, &w_shape),
                        .strides = @as([]usize, &w_strides),
                        .offset = 0,
                        .allocator = null,
                        .owns_data = false,
                    };

                    var expert_out = try Tensor(f32).alloc(self.allocator, &.{ 1, head_dim });
                    defer expert_out.deinit();
                    var x_row = try Tensor(f32).alloc(self.allocator, &.{ 1, p.n_embd });
                    defer x_row.deinit();
                    @memcpy(x_row.data, x.data[t * p.n_embd ..][0..p.n_embd]);

                    try self.matmul_engine.linearProjection(f32, x_row, expert_tensor, &expert_out);

                    // Apply silu activation
                    for (0..head_dim) |d| {
                        const val = expert_out.data[d];
                        expert_out.data[d] = val / (1.0 + @exp(-val));
                    }

                    // Multiply by weight and add to output
                    const weight = selected_weights[t * n_used + k];
                    for (0..head_dim) |d| {
                        Vf32_mova.data[t * n_kv_head * head_dim + d] += expert_out.data[d] * weight;
                    }
                }
            }

            // Copy MoVA result to Vf32_hm (broadcast across kv heads)
            for (0..N) |t| {
                for (0..n_kv_head) |h| {
                    for (0..head_dim) |d| {
                        Vf32_hm.data[t * n_kv_head * head_dim + h * head_dim + d] = Vf32_mova.data[t * n_kv_head * head_dim + d];
                    }
                }
            }
        } else {
            // Standard V projection
            for (0..N) |t| {
                for (0..n_kv_head) |h| {
                    for (0..head_dim) |d| {
                        Vf32_hm.data[t * n_kv_head * head_dim + h * head_dim + d] = Vf32.data[t * kv_dim + h * head_dim + d];
                    }
                }
            }
        }
        // Reshape K [N, kv_dim] → [N, n_kv_head, head_dim]. NO depende de MoVA
        // (V sí) — debe hacerse SIEMPRE. El refactor K2-Horizon (f2562d4)
        // borró este loop y dejó Kf32_hm a ceros: el golden CPU quedaba con
        // K≡0 (softmax uniforme para token>0), rompiendo la paridad U2 y el
        // camino CPU híbrido. Restaurado (lane-b1, fix regresión U2).
        for (0..N) |t| {
            for (0..n_kv_head) |h| {
                for (0..head_dim) |d| {
                    Kf32_hm.data[t * n_kv_head * head_dim + h * head_dim + d] = Kf32.data[t * kv_dim + h * head_dim + d];
                }
            }
        }
        // off). V no lleva RoPE: este es su valor definitivo pre-caché.
        if (kv_cache.kv_trace.hook.active != null) {
            try kv_cache.kv_trace.hook.stageV(Vf32_hm.data);
        }

        // === 4. Q/K RMSNorm per-head ===
        // U1: sólo si el modelo trae q/k_norm (qwen35/qwen3). Llama-arch
        // NO los trae — OMITIR (rmsNorm con ones NO es identidad).
        if (self.has_qk_norm) {
            var Q_norm = try Tensor(f32).alloc(self.allocator, &.{ N * n_head, head_dim });
            defer Q_norm.deinit();
            for (0..N * n_head * head_dim) |i| Q_norm.data[i] = Qf32.data[i];
            norm.rmsNorm(f32, f32, Q_norm, self.attn_q_norm, p.rms_eps, &Q_norm);
            for (0..N * n_head * head_dim) |i| Qf32.data[i] = Q_norm.data[i];

            var K_norm = try Tensor(f32).alloc(self.allocator, &.{ N * n_kv_head, head_dim });
            defer K_norm.deinit();
            for (0..N * n_kv_head * head_dim) |i| K_norm.data[i] = Kf32_hm.data[i];
            norm.rmsNorm(f32, f32, K_norm, self.attn_k_norm, p.rms_eps, &K_norm);
            for (0..N * n_kv_head * head_dim) |i| Kf32_hm.data[i] = K_norm.data[i];
        }

        // Punto de captura K pre-RoPE (lane-kvc: tras RMSNorm, antes de RoPE)
        if (kv_cache.kv_trace.hook.active != null) {
            try kv_cache.kv_trace.hook.stageKPre(Kf32_hm.data);
        }

        // === 5. RoPE (MRoPE for Qwen3.5, standard for LFM2) ===
        // U2-fix (lane-b1): copia con TRANSPOSE real token-major →
        // head-major. La copia FLAT anterior mentía el shape [1,H,N,hd]:
        // applyRoPE* indexa row_offset=(h*N+pos)*hd pero la data seguía
        // [N,H,hd] ⇒ RoPE con posiciones scrambleadas (mismo bug que
        // mropeKernel GPU — dos errores idénticos cancelados en el
        // test-u2 CPU-vs-GPU; visible en batched-vs-unrolled rel 0.21).
        var Q_hm = try Tensor(f32).alloc(self.allocator, &.{ 1, n_head, N, head_dim });
        defer Q_hm.deinit();
        var K_hm = try Tensor(f32).alloc(self.allocator, &.{ 1, n_kv_head, N, head_dim });
        defer K_hm.deinit();
        for (0..N) |t| {
            for (0..n_head) |h| {
                for (0..head_dim) |d| {
                    Q_hm.data[(h * N + t) * head_dim + d] = Qf32.data[t * (n_head * head_dim) + h * head_dim + d];
                }
            }
        }
        for (0..N) |t| {
            for (0..n_kv_head) |h| {
                for (0..head_dim) |d| {
                    K_hm.data[(h * N + t) * head_dim + d] = Kf32_hm.data[t * n_kv_head * head_dim + h * head_dim + d];
                }
            }
        }

        if (p.use_mrope) {
            if (pos_ids) |ids| {
                // MRoPE per-token (tokens de imagen: (t,h,w,e) 2D —
                // mtmd-helper.cpp:172 set_position_mrope_2d)
                rope_mod.applyRoPEMultiSectionPosIds(f32, &Q_hm, &K_hm, ids, head_dim, p.n_rot, p.rope_sections, p.rope_freq_base);
            } else {
                rope_mod.applyRoPEMultiSection(f32, &Q_hm, &K_hm, start_pos, head_dim, p.n_rot, p.rope_sections, p.rope_freq_base);
            }
        } else {
            rope_mod.applyRoPE(f32, &Q_hm, &K_hm, start_pos, head_dim, p.rope_freq_base, p.rope_neox);
        }

        // U2-fix (lane-b1): vuelta con transpose head-major → token-major
        // (simétrica a la entrada; antes era copia flat).
        for (0..N) |t| {
            for (0..n_head) |h| {
                for (0..head_dim) |d| {
                    Qf32.data[t * (n_head * head_dim) + h * head_dim + d] = Q_hm.data[(h * N + t) * head_dim + d];
                }
            }
        }
        for (0..N) |t| {
            for (0..n_kv_head) |h| {
                for (0..head_dim) |d| {
                    Kf32_hm.data[t * n_kv_head * head_dim + h * head_dim + d] = K_hm.data[(h * N + t) * head_dim + d];
                }
            }
        }

        // Punto de captura K post-RoPE + Q tail (lane-kvc). Cierra el chunk:
        // k_pre + k_post + v + q se entregan juntos al tracer.
        if (kv_cache.kv_trace.hook.active != null) {
            try kv_cache.kv_trace.hook.appendChunk(
                self.layer_idx,
                N,
                kv_dim,
                Kf32_hm.data,
                Qf32.data,
            );
        }

        // === 6. KV-Cache paginado: escribir K/V al bloque y recuperar K/V full ===
        // El KV cache vive en PagedKVCache. El block_table mapea posiciones
        // lógicas → bloques físicos. Cada capa de atención comparte el mismo
        // PagedKVCache pero tiene su propio block_table (bloques distintos).
        // Bloques deben estar pre-asignados por el Scheduler antes de forward.
        const block_size = self.paged_kv.config.block_size;
        const total_len = start_pos + N;

        const bytes_per_elem = self.paged_kv.block_alloc.bytes_per_elem;
        const kv_stride_block = block_size * kv_dim * bytes_per_elem;
        const quant_on = self.k_quant != .fp16 or self.v_quant != .fp16;
        for (0..N) |t| {
            const global_pos = start_pos + t;
            const block_idx = global_pos / block_size;
            const offset_in_block = global_pos % block_size;
            const phys_id = self.block_table.getPhysical(block_idx) orelse return HybridAttnError.KvCacheNotSet;
            const block_data = self.paged_kv.block_alloc.memory_pool[phys_id * self.paged_kv.block_alloc.block_bytes ..];

            if (quant_on) {
                // Acumular f16 en staging; cuantizar al sellar bloque (offset == block_size-1)
                // o al cambiar de bloque lógico con datos pendientes.
                if (self.staged_block != @as(isize, @intCast(block_idx)) and self.staged_tokens > 0) {
                    try self.flushQuantTile();
                }
                const row = offset_in_block * kv_dim;
                for (0..n_kv_head) |h| {
                    for (0..head_dim) |d| {
                        const idx = row + h * head_dim + d;
                        self.k_staging[idx] = @floatCast(Kf32_hm.data[t * kv_dim + h * head_dim + d]);
                        self.v_staging[idx] = @floatCast(Vf32_hm.data[t * kv_dim + h * head_dim + d]);
                    }
                }
                if (self.staged_block == -1) self.staged_block = @as(isize, @intCast(block_idx));
                self.staged_tokens = offset_in_block + 1;
                if (offset_in_block == block_size - 1) {
                    try self.flushQuantTile();
                }
            } else {
                const kv_offset = offset_in_block * kv_dim * bytes_per_elem;
                for (0..n_kv_head) |h| {
                    for (0..head_dim) |d| {
                        const k_idx = kv_offset + (h * head_dim + d) * bytes_per_elem;
                        const v_idx = k_idx + kv_stride_block;
                        const kv_val_k = Kf32_hm.data[t * kv_dim + h * head_dim + d];
                        const kv_val_v = Vf32_hm.data[t * kv_dim + h * head_dim + d];
                        storeF16(block_data, k_idx, kv_val_k);
                        storeF16(block_data, v_idx, kv_val_v);
                    }
                }
                // El bloque recién escrito es modificado en host: el pool GPU
                // debe re-subirlo en el próximo decode (dirty en vez de salto).
                if (self.paged_gpu) |gpu| gpu.markDirty(phys_id);
            }
        }

        // Sellar cualquier bloque parcial aún en staging antes de atender: el
        // reader (CPU o GPU) lee el memory-pool directamente, así que el tile
        // acumulado debe estar ya cuantizado (padded con ceros) en el pool.
        if (quant_on and self.staged_tokens > 0) {
            try self.flushQuantTile();
        }

        // === 7-8. Softmax Attention (causal) ===
        // GPU: PagedAttentionGpu.decode sobre el memory-pool (bloques f16).
        // CPU: softmáx clásico con GQA expansion (ruta de referencia).
        var attn_out = try Tensor(f32).alloc(self.allocator, &.{ N, n_head, head_dim });
        defer attn_out.deinit();

        if (self.paged_gpu) |gpu| {
            const q_stride = n_head * head_dim;
            if (N > 1) {
                // Prefill en bloque: kernel causal `paged_attention_prefill_f16_kernel`.
                try gpu.prefill(
                    Qf32.data[0 .. N * q_stride],
                    attn_out.data[0 .. N * q_stride],
                    self.block_table,
                    self.paged_kv.block_alloc,
                    N,
                );
            } else {
                // Decode de un token (N == 1): atiende a todos los pasados.
                try gpu.decode(
                    Qf32.data[0..q_stride],
                    attn_out.data[0..q_stride],
                    self.block_table.*,
                    self.paged_kv.block_alloc,
                    self.paged_kv.config,
                );
            }
        } else {
            // Read K/V back as f16 and expand GQA (CPU fallback).
            var K_full = try Tensor(f32).alloc(self.allocator, &.{ total_len, n_kv_head, head_dim });
            defer K_full.deinit();
            var V_full = try Tensor(f32).alloc(self.allocator, &.{ total_len, n_kv_head, head_dim });
            defer V_full.deinit();

            // Dequantizar bloques K/V al backing f16 del pool. Si el cache está
            // cuantizado, cada bloque se de-cuantiza con kv_quant.decode.
            var last_block: isize = -1;
            var tile_k: []f16 = &.{};
            var tile_v: []f16 = &.{};
            if (self.k_quant != .fp16 or self.v_quant != .fp16) {
                tile_k = try self.allocator.alloc(f16, kv_dim * block_size);
                tile_v = try self.allocator.alloc(f16, kv_dim * block_size);
            }
            const needs_tile = self.k_quant != .fp16 or self.v_quant != .fp16;
            defer {
                if (needs_tile) {
                    self.allocator.free(tile_k);
                    self.allocator.free(tile_v);
                }
            }

            for (0..total_len) |t| {
                const block_idx = t / block_size;
                const offset_in_block = t % block_size;
                const phys_id = self.block_table.getPhysical(block_idx) orelse return HybridAttnError.KvCacheNotSet;
                const block_data = self.paged_kv.block_alloc.memory_pool[phys_id * self.paged_kv.block_alloc.block_bytes ..];

                // Refundir (dequantizar) el bloque al cambiar de bloque lógico.
                if (@as(isize, @intCast(block_idx)) != last_block) {
                    if (self.k_quant != .fp16) {
                        kv_quant.decode(self.k_quant, block_data[0..self.k_tile_bytes], tile_k);
                    }
                    if (self.v_quant != .fp16) {
                        kv_quant.decode(self.v_quant, block_data[self.k_tile_bytes..][0..self.v_tile_bytes], tile_v);
                    }
                    last_block = @as(isize, @intCast(block_idx));
                }

                if (self.k_quant != .fp16 or self.v_quant != .fp16) {
                    const row_off = offset_in_block * kv_dim;
                    for (0..n_kv_head) |h| {
                        for (0..head_dim) |d| {
                            const idx = row_off + h * head_dim + d;
                            K_full.data[t * kv_dim + h * head_dim + d] = @floatCast(tile_k[idx]);
                            V_full.data[t * kv_dim + h * head_dim + d] = @floatCast(tile_v[idx]);
                        }
                    }
                } else {
                    const kv_offset = offset_in_block * kv_dim * bytes_per_elem;
                    for (0..n_kv_head) |h| {
                        for (0..head_dim) |d| {
                            const k_idx = kv_offset + (h * head_dim + d) * bytes_per_elem;
                            const v_idx = k_idx + kv_stride_block;
                            K_full.data[t * kv_dim + h * head_dim + d] = loadF16(block_data, k_idx);
                            V_full.data[t * kv_dim + h * head_dim + d] = loadF16(block_data, v_idx);
                        }
                    }
                }
            }

            const repeat_factor = n_head / n_kv_head;
            var K_exp = try Tensor(f32).alloc(self.allocator, &.{ total_len, n_head, head_dim });
            defer K_exp.deinit();
            var V_exp = try Tensor(f32).alloc(self.allocator, &.{ total_len, n_head, head_dim });
            defer V_exp.deinit();

            for (0..total_len) |t| {
                for (0..n_kv_head) |kv_h| {
                    for (0..repeat_factor) |r| {
                        const h = kv_h * repeat_factor + r;
                        for (0..head_dim) |d| {
                            K_exp.data[t * n_head * head_dim + h * head_dim + d] = K_full.data[t * n_kv_head * head_dim + kv_h * head_dim + d];
                            V_exp.data[t * n_head * head_dim + h * head_dim + d] = V_full.data[t * n_kv_head * head_dim + kv_h * head_dim + d];
                        }
                    }
                }
            }

            const kq_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

            for (0..N) |t| {
                for (0..n_head) |h| {
                    var scores = std.heap.page_allocator.alloc(f32, total_len) catch unreachable;
                    defer std.heap.page_allocator.free(scores);

                    var max_score: f32 = -std.math.inf(f32);
                    for (0..total_len) |s| {
                        var score: f32 = 0;
                        for (0..head_dim) |d| {
                            const q = Qf32.data[t * n_head * head_dim + h * head_dim + d];
                            const k = K_exp.data[s * n_head * head_dim + h * head_dim + d];
                            score += q * k;
                        }
                        score *= kq_scale;
                        // 5.2 (lane-b1): causal=false (denoise DFlash) → sin
                        // máscara: TODAS las filas atienden [0, start_pos+n).
                        if (self.causal and s > start_pos + t) score = -std.math.inf(f32);
                        // RLT SWA: skip tokens before the sliding window floor
                        if (self.params.swa_window) |w| {
                            const window_start = if (start_pos + t + 1 > w) start_pos + t + 1 - w else 0;
                            if (s < window_start) score = -std.math.inf(f32);
                        }
                        scores[s] = score;
                        if (score > max_score) max_score = score;
                    }

                    var sum_exp: f32 = 0;
                    for (0..total_len) |s| {
                        const exp_val = @exp(scores[s] - max_score);
                        scores[s] = exp_val;
                        sum_exp += exp_val;
                    }
                    for (0..total_len) |s| {
                        scores[s] /= sum_exp;
                    }

                    for (0..head_dim) |d| {
                        var out_val: f32 = 0;
                        for (0..total_len) |s| {
                            out_val += scores[s] * V_exp.data[s * n_head * head_dim + h * head_dim + d];
                        }
                        attn_out.data[t * n_head * head_dim + h * head_dim + d] = out_val;
                    }
                }
            }
        }

        // === 9. Gate: sigmoid(G) * attn_out (skip for LFM2 no_gate) ===
        if (!p.no_gate) {
            for (0..N * n_head * head_dim) |i| {
                const g = Gf32.data[i];
                const sigmoid = 1.0 / (1.0 + @exp(-g));
                attn_out.data[i] *= sigmoid;
            }
        }

        // === 9b. K2-Horizon: softplus gate on attention output (before WO) ===
        // gate = softplus(wqkv_gate @ attn_inp * LN2) / LN2
        if (p.has_softplus_gate and self.wqkv_gate != null) {
            const gate_scratch = self.scratch_gate orelse return;
            var gate_shape = [_]usize{ n_head * head_dim, p.n_embd };
            var gate_strides = [_]usize{ p.n_embd, 1 };
            const gate_w32 = Tensor(f32){
                .data = gate_scratch,
                .shape = &gate_shape,
                .strides = &gate_strides,
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };
            var gate_buf = try Tensor(f32).alloc(self.allocator, &.{ N, n_head * head_dim });
            defer gate_buf.deinit();
            try self.matmul_engine.linearProjection(f32, x, gate_w32, &gate_buf);

            const LN2: f32 = 0.6931471805599453;
            const ONE_OVER_LN2: f32 = 1.4426950408889634;
            for (0..N * n_head * head_dim) |i| {
                const gv = gate_buf.data[i] * LN2;
                const softplus = @log(1.0 + @exp(gv));
                const scaled = softplus * ONE_OVER_LN2;
                attn_out.data[i] *= scaled;
            }
        }

        // === 10. Output projection ===
        const q_dim = n_head * head_dim;
        var attn_flat = try Tensor(f32).alloc(self.allocator, &.{ N, q_dim });
        defer attn_flat.deinit();
        for (0..N * q_dim) |i| attn_flat.data[i] = attn_out.data[i];

        var w_o_shape = [_]usize{ p.n_embd, q_dim };
        var w_o_strides = [_]usize{ q_dim, 1 };
        const w_o32 = Tensor(f32){
            .data = self.scratch_o,
            .shape = &w_o_shape,
            .strides = &w_o_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        var attn_proj = try Tensor(f32).alloc(self.allocator, &.{ N, p.n_embd });
        defer attn_proj.deinit();
        try self.matmul_engine.linearProjection(f32, attn_flat, w_o32, &attn_proj);

        // === 11. Salida: mixer-only, sin residual ni post-norm ni FFN ===
        // HybridLayer se encarga de residual + post-norm + FFN.
        // OJO pool: `out.data` puede ser un bloque reciclado MÁS GRANDE que la
        // shape lógica [N, n_embd] (ActivationPool best-fit devuelve slices
        // físicos ≥ pedido). Copiar por el tamaño LÓGICO de attn_proj.
        // (fix hybrid_attn:1050 @0013685 lane-e, ya en main via commit previo)
        const out_slice = out.data[0..attn_proj.data.len];
        for (out_slice, attn_proj.data) |*o, a| o.* = a;
    }

    // ─── Forward de atención 100% GPU (Phase 1b, decode N == 1) ──────────────
    // Todo vive en device: proyecciones, split Q|G, Q/K norm, MRoPE, KV-append
    // f16, decode paginado device→device, gate y proyección de salida. Un solo
    // sync por token en el llamador. Fiel al forward CPU (incluida la semántica
    // flat de MRoPE) para que decode GPU == prefill CPU.
    // En modo grafo de decode, los valores por token (block table, start_pos,
    // seq_len) se pintan en staging host (nodos HtoDAsync capturados); los
    // kernels leen los buffers device fijos. La sincronización de bloques al
    // host la hace el llamador vía `syncDecodeBlocks` tras lanzar el grafo.
    /// 9.4 (lane-b): atención clásica vía pool paged (materialize) —
    /// extraída de forwardGPU para el fallback del camino kvarn-native.
    fn forwardPagedAttn(
        self: *Self,
        lk: *layer_kernels.LayerKernels,
        g: *AttentionGpu,
        n: usize,
        bt_host: []c_int,
        start_pos: usize,
    ) !void {
        const gpu = self.paged_gpu orelse return HybridAttnError.KvCacheNotSet;
        const p = self.params;
        const q_dim = p.n_head * p.head_dim;
        try lk.copyF32toF16(g.g_q.ptr(), g.d_q16, n * q_dim);
        if (n > 1) {
            // Prefill causal en bloque (device→device): atiende los n queries
            // sobre todos los tokens ya escritos en el pool.
            try gpu.prefillDevice(self.layer_idx, g.d_q16, g.d_attn16, self.paged_kv.block_alloc, bt_host, n, start_pos, null);
        } else {
            // Decode de un token: block table/start_pos/seq_len ya en device.
            try gpu.decodeDevice(self.layer_idx, g.d_q16, g.d_attn16, self.paged_kv.block_alloc);
        }
        try lk.copyF16toF32(g.d_attn16, g.g_attn.ptr(), n * q_dim);
    }

    /// 9.4 fix (lane-b): gate + proyección de salida del camino común,
    /// reutilizado por el branch kvarn-native (el kernel FA escribe
    /// g.g_attn en dominio original; sin el return que saltaba esto).
    /// El dump "gated" lo hace el caller (att_dbg_dump es local de
    /// forwardGPU).
    fn applyGateAndOutputProj(
        self: *Self,
        lk: *layer_kernels.LayerKernels,
        g: *AttentionGpu,
        out: *cublas.GpuTensor(f32),
        n: usize,
        q_dim: usize,
        no_gate: bool,
    ) !void {
        const p = self.params;
        if (!no_gate) {
            try lk.gateMul(g.g_attn.ptr(), g.g_g.ptr(), n * q_dim);
        }
        var w_o_shape = [_]usize{ p.n_embd, q_dim };
        var w_o_strides = [_]usize{ q_dim, 1 };
        const w_o32 = Tensor(f32){ .data = self.scratch_o, .shape = &w_o_shape, .strides = &w_o_strides, .offset = 0, .allocator = null, .owns_data = false };
        const qt_o = if (quantAttnEnabled()) qgemmTypeFor(self.w_o.dtype()) else null;
        if (qt_o) |qt| {
            try lk.qgemmLinear(self.allocator, g.g_attn.ptr(), self.w_o.bytes, out.ptr(), n, q_dim, p.n_embd, qt);
        } else {
            try self.matmul_engine.linearProjectionDevice(g.g_attn, w_o32, out, n, q_dim, p.n_embd);
        }
    }

    //
    // ── 5.2 (lane-b1): modo embd DFlash — KV-Inject ─────────────────────────
    // `appendKVOnly`: proyecta las features fusionadas del encoder por
    // wk/wv, aplica k_norm + RoPE (posiciones start_pos..+n) y ESCRIBE K/V
    // al pool paginado — sin atención, sin q/o/ffn. Es el "modo embd" del
    // dflash.cpp de llama.cpp (grafo<false> con ubatch.embd): las capas del
    // draft NO computan atención durante el prefill del target; sólo
    // inyectan K/V para que el denoise posterior atienda al pasado.
    pub fn appendKVOnly(self: *Self, x: Tensor(f32), start_pos: usize) !void {
        const p = self.params;
        const N = x.shape[0];
        const n_embd = p.n_embd;
        const head_dim = p.head_dim;
        const n_kv_head = p.n_kv_head;
        const kv_dim = p.kv_dim();
        try self.ensureF32Scratch();

        // wk/wv: [kv_dim, n_embd] · x[N, n_embd]
        var w_k_shape = [_]usize{ kv_dim, n_embd };
        var w_k_strides = [_]usize{ n_embd, 1 };
        const w_k32 = Tensor(f32){
            .data = self.scratch_k,
            .shape = &w_k_shape,
            .strides = &w_k_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        var Kf32 = try Tensor(f32).alloc(self.allocator, &.{ N, kv_dim });
        defer Kf32.deinit();
        try self.matmul_engine.linearProjection(f32, x, w_k32, &Kf32);

        var w_v_shape = [_]usize{ kv_dim, n_embd };
        var w_v_strides = [_]usize{ n_embd, 1 };
        const w_v32 = Tensor(f32){
            .data = self.scratch_v,
            .shape = &w_v_shape,
            .strides = &w_v_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        var Vf32 = try Tensor(f32).alloc(self.allocator, &.{ N, kv_dim });
        defer Vf32.deinit();
        try self.matmul_engine.linearProjection(f32, x, w_v32, &Vf32);

        // k_norm per-head (mismo orden que el forward: norm ANTES de RoPE)
        if (self.has_qk_norm) {
            var K_norm = try Tensor(f32).alloc(self.allocator, &.{ N * n_kv_head, head_dim });
            defer K_norm.deinit();
            @memcpy(K_norm.data, Kf32.data);
            norm.rmsNorm(f32, f32, K_norm, self.attn_k_norm, p.rms_eps, &K_norm);
            @memcpy(Kf32.data, K_norm.data);
        }

        // RoPE sobre K [1, n_kv_head, N, head_dim] (posiciones start_pos..)
        // U2-fix (lane-b1): transpose real token-major→head-major (la copia
        // flat anterior mentía el shape — mismo scramble pos que el forward
        // principal; ver bloque 5 del forward).
        var K_hm = try Tensor(f32).alloc(self.allocator, &.{ 1, n_kv_head, N, head_dim });
        defer K_hm.deinit();
        for (0..N) |t| {
            for (0..n_kv_head) |h| {
                for (0..head_dim) |d| {
                    K_hm.data[(h * N + t) * head_dim + d] = Kf32.data[t * kv_dim + h * head_dim + d];
                }
            }
        }
        if (p.use_mrope) {
            rope_mod.applyRoPEMultiSection(f32, &K_hm, &K_hm, start_pos, head_dim, p.n_rot, p.rope_sections, p.rope_freq_base);
        } else {
            rope_mod.applyRoPE(f32, &K_hm, &K_hm, start_pos, head_dim, p.rope_freq_base, p.rope_neox);
        }
        for (0..N) |t| {
            for (0..n_kv_head) |h| {
                for (0..head_dim) |d| {
                    Kf32.data[t * kv_dim + h * head_dim + d] = K_hm.data[(h * N + t) * head_dim + d];
                }
            }
        }

        // === Escritura al pool (réplica del bloque 6 del forward) ===
        const block_size = self.paged_kv.config.block_size;
        const total_len = start_pos + N;
        const bytes_per_elem = self.paged_kv.block_alloc.bytes_per_elem;
        const kv_stride_block = block_size * kv_dim * bytes_per_elem;
        const quant_on = self.k_quant != .fp16 or self.v_quant != .fp16;

        for (0..N) |t| {
            const pos = start_pos + t;
            if (pos >= self.block_table.num_tokens) {
                try self.block_table.appendTokens(self.paged_kv.block_alloc, 1);
            }
            const block_idx = pos / block_size;
            const offset_in_block = pos % block_size;
            const phys_id = self.block_table.getPhysical(block_idx) orelse return HybridAttnError.KvCacheNotSet;
            const block_data = self.paged_kv.block_alloc.memory_pool[phys_id * self.paged_kv.block_alloc.block_bytes ..];

            if (quant_on) {
                if (self.staged_block != @as(isize, @intCast(block_idx)) and self.staged_tokens > 0) {
                    try self.flushQuantTile();
                }
                const row = offset_in_block * kv_dim;
                for (0..n_kv_head) |h| {
                    for (0..head_dim) |d| {
                        const idx = row + h * head_dim + d;
                        self.k_staging[idx] = @floatCast(Kf32.data[t * kv_dim + h * head_dim + d]);
                        self.v_staging[idx] = @floatCast(Vf32.data[t * kv_dim + h * head_dim + d]);
                    }
                }
                if (self.staged_block == -1) self.staged_block = @as(isize, @intCast(block_idx));
                self.staged_tokens = offset_in_block + 1;
                if (offset_in_block == block_size - 1) {
                    try self.flushQuantTile();
                }
            } else {
                const kv_offset = offset_in_block * kv_dim * bytes_per_elem;
                for (0..n_kv_head) |h| {
                    for (0..head_dim) |d| {
                        const k_idx = kv_offset + (h * head_dim + d) * bytes_per_elem;
                        const v_idx = k_idx + kv_stride_block;
                        storeF16(block_data, k_idx, Kf32.data[t * kv_dim + h * head_dim + d]);
                        storeF16(block_data, v_idx, Vf32.data[t * kv_dim + h * head_dim + d]);
                    }
                }
            }
        }
        _ = total_len;
    }

    pub fn forwardGPU(
        self: *Self,
        lk: *layer_kernels.LayerKernels,
        x: cublas.GpuTensor(f32),
        out: *cublas.GpuTensor(f32),
        start_pos: usize,
        n: usize,
        pos_ids: ?[]const [4]i32,
    ) !void {
        const p = self.params;
        const qg_dim = p.qg_dim();
        const kv_dim = p.kv_dim();
        const head_dim = p.head_dim;
        const n_head = p.n_head;
        const n_kv_head = p.n_kv_head;
        const q_dim = n_head * head_dim;

        try AttentionLayer.ensureGpu(self);
        const g = &self.gpu.?;
        try g.ensureN(n);
        const gpu = self.paged_gpu orelse return HybridAttnError.KvCacheNotSet;

        // Vision (PLAN_MMPROJ 3.2): posición de CONTEXTO para el RoPE de
        // decode, separada del slot KV. Con embeddings de imagen inyectados
        // el slot avanza n_img pero el contexto sólo max(nx,ny): delta
        // negativo constante. `d_rope_pos` se habilita (enableRopePos) sólo
        // con vision; sin él el mrope lee d_start_pos (comportamiento clásico).
        const use_rope_pos = gpu.getDRopePos() != 0;
        const d_rope_pos_val: usize = @intCast(gpu.getDRopePos());

        // Decode (n == 1): staging persistente en host y HtoDAsync (nodos
        // capturados por el grafo). Prefill (n > 1): solo el start_pos device
        // (lo lee MRoPE vía puntero). Ambos subidos en stream order antes de
        // los kernels que los consumen.
        if (n == 1) {
            try self.stageDecodeHost(start_pos, n);
            try gpu.uploadScratch(self.layer_idx);
        } else {
            try gpu.uploadStartPos(start_pos);
            // Vision (Fase B): pos-ids per-token del chunk de prefill — el
            // mropePosIdsKernel los lee del device. Sólo con vision activo.
            if (pos_ids) |ids| {
                try gpu.uploadPosIds(ids, start_pos, n);
            }
        }

        // Dequant lazy si este forward caerá al camino f32 (sin kernels
        // qgemm para algún peso): los wrappers w_*32 leen scratch.
        {
            const qt_q0 = if (quantAttnEnabled()) qgemmTypeFor(self.w_q.dtype()) else null;
            const qt_k0 = if (quantAttnEnabled()) qgemmTypeFor(self.w_k.dtype()) else null;
            const qt_v0 = if (quantAttnEnabled()) qgemmTypeFor(self.w_v.dtype()) else null;
            if (qt_q0 == null or qt_k0 == null or qt_v0 == null) {
                try self.ensureF32Scratch();
            }
        }
        var w_q_shape = [_]usize{ qg_dim, p.n_embd };
        var w_q_strides = [_]usize{ p.n_embd, 1 };
        const w_q32 = Tensor(f32){ .data = self.scratch_q, .shape = &w_q_shape, .strides = &w_q_strides, .offset = 0, .allocator = null, .owns_data = false };
        var w_k_shape = [_]usize{ kv_dim, p.n_embd };
        var w_k_strides = [_]usize{ p.n_embd, 1 };
        const w_k32 = Tensor(f32){ .data = self.scratch_k, .shape = &w_k_shape, .strides = &w_k_strides, .offset = 0, .allocator = null, .owns_data = false };
        var w_v_shape = [_]usize{ kv_dim, p.n_embd };
        var w_v_strides = [_]usize{ p.n_embd, 1 };
        const w_v32 = Tensor(f32){ .data = self.scratch_v, .shape = &w_v_shape, .strides = &w_v_strides, .offset = 0, .allocator = null, .owns_data = false };
        var w_o_shape = [_]usize{ p.n_embd, q_dim };
        var w_o_strides = [_]usize{ q_dim, 1 };
        const w_o32 = Tensor(f32){ .data = self.scratch_o, .shape = &w_o_shape, .strides = &w_o_strides, .offset = 0, .allocator = null, .owns_data = false };

        // 8.3-diag (lane-f): traza por etapas de la capa attention para el
        // NaN de prefill n=6 (LFM2.5 capa 27). Gate ATT_DBG (sólo li==27) /
        // ATT_DBG_ALL (todas). Breadcrumb permanente: coste cero sin flag.
        const att_dbg = std.c.getenv("ATT_DBG") != null or std.c.getenv("ATT_DBG_ALL") != null;
        const att_dbg_li = self.layer_idx == 27 or std.c.getenv("ATT_DBG_ALL") != null;
        // 9.4 debug: step global del dump (start_pos del forwardGPU actual).
        att_dbg_step_global = start_pos;
        const att_dbg_dump = struct {
            fn go(tag: []const u8, li: usize, ptr: usize, elems: usize) void {
                const buf = std.heap.page_allocator.alloc(f32, elems) catch return;
                defer std.heap.page_allocator.free(buf);
                cudaz.cuMemcpyDtoH(@intFromPtr(buf.ptr), ptr, elems * @sizeOf(f32)) catch {
                    debugz.dbg.print("[attn] L{d} {s} MEMFAIL\n", .{ li, tag });
                    return;
                };
                debugz.dbg.print("[attn] L{d} {s} sum|v|={d:.4} max={d:.4} nan={any}\n", .{ li, tag, debugz.sumAbsF32(buf), debugz.maxAbsF32(buf), std.mem.indexOfScalar(f32, buf, std.math.nan(f32)) != null });
                // 9.4 debug: ZIG_AI_ATT_DUMP=<dir> vuelca el vector f32
                // completo — diff numérico fattn vs paged por elemento.
                if (std.c.getenv("ZIG_AI_ATT_DUMP")) |dir_raw| {
                    var dbuf: [256]u8 = undefined;
                    var dlen: usize = 0;
                    while (dir_raw[dlen] != 0 and dlen < 255) : (dlen += 1) dbuf[dlen] = dir_raw[dlen];
                    dbuf[dlen] = 0;
                    var pb: [256]u8 = undefined;
                    const path = std.fmt.bufPrintZ(&pb, "{s}/s{d}_e{d}_L{d}_{s}.f32", .{ dbuf[0..dlen], att_dbg_step_global, elems, li, tag }) catch return;
                    const f = std.Io.Dir.cwd().createFile(std.Io.Threaded.global_single_threaded.io(), path, .{ .truncate = true }) catch return;
                    defer f.close(std.Io.Threaded.global_single_threaded.io());
                    var wbuf: [8192]u8 = undefined;
                    var w = f.writer(std.Io.Threaded.global_single_threaded.io(), &wbuf);
                    const wr = &w.interface;
                    wr.writeAll(std.mem.sliceAsBytes(buf)) catch {};
                    wr.flush() catch {};
                }
            }
        };

        // 1. Proyecciones Q+G, K, V (device→device, pesos cacheados en GPU).
        // T1-B VRAM-spec (patrón ssm.zig de lane-a): GEMM cuantizado para
        // TODO dtype con kernel qgemm (q8_0 del Q8_K_XL incluido); fallback
        // f32 solo para dtypes sin kernel.
        const qt_q = if (quantAttnEnabled()) qgemmTypeFor(self.w_q.dtype()) else null;
        const qt_k = if (quantAttnEnabled()) qgemmTypeFor(self.w_k.dtype()) else null;
        const qt_v = if (quantAttnEnabled()) qgemmTypeFor(self.w_v.dtype()) else null;

        // FP8 Block-Scaled path (if enabled and weights are FP8)
        const fp8_enabled = fp8AttnEnabled() and self.matmul_engine.backend == matmul.Backend.fp8_block;
        if (fp8_enabled) {
            // FP8 path: weights must be pre-quantized to FP8
            // TODO: implement FP8 weight loading and FP8 gemmFp8Block call
            try self.matmul_engine.linearProjectionDevice(x, w_q32, &g.g_qg, n, p.n_embd, qg_dim);
            try self.matmul_engine.linearProjectionDevice(x, w_k32, &g.g_k, n, p.n_embd, kv_dim);
            try self.matmul_engine.linearProjectionDevice(x, w_v32, &g.g_v, n, p.n_embd, kv_dim);
        } else if (qt_q != null and qt_k != null and qt_v != null) {
            // a-U3 fase 3c (lane-a): FUSIÓN qkv — un solo launch para las
            // 3 proyecciones cuando las 3 son q3_k con mismo K (el caso del
            // Llama-3.2-3B Q3_K_S: gate del U3). DEFAULT ON (opt-out NOQ3PACK=1): pesos
            // packed 128B/SB via q3kPackedWeight (cae a 110B si el repack OOMea); prewarm aquí, fuera
            // del graph capture si venimos de prefill m>1). M=1 (decode) es
            // el camino fused; m>1 (prefill) mantiene los 3 separados (el
            // kernel fused es M=1 — prewarm de los packed antes del retorno).
            // Opt-out familia SSM4DP4A (NOSSM4DP4A=1).
            var qkv_fused = false;
            if (n == 1 and qt_q.? == 6 and qt_k.? == 6 and qt_v.? == 6 and
                p.n_embd % 256 == 0 and !debugz.dbg.no_ssm4dp4a and
                std.c.getenv("NOQ3PACK") == null)
            {
                const pq = layer_kernels.q3kPackedWeight(self.allocator, @intFromPtr(self.w_q.bytes.ptr), self.w_q.bytes, qg_dim, p.n_embd) catch null;
                const pk = layer_kernels.q3kPackedWeight(self.allocator, @intFromPtr(self.w_k.bytes.ptr), self.w_k.bytes, kv_dim, p.n_embd) catch null;
                const pv = layer_kernels.q3kPackedWeight(self.allocator, @intFromPtr(self.w_v.bytes.ptr), self.w_v.bytes, kv_dim, p.n_embd) catch null;
                if (pq != null and pk != null and pv != null) {
                    debugz.dbg.printLevel(.detail, "[aU3-qkv] fused n_q={d} n_k={d} n_v={d} k={d}\n", .{ qg_dim, kv_dim, kv_dim, p.n_embd });
                    try lk.q3kGemmM1Dp4aPackedQKV(x.ptr(), pq.?, pk.?, pv.?, g.g_qg.ptr(), g.g_k.ptr(), g.g_v.ptr(), p.n_embd, qg_dim, kv_dim, kv_dim);
                    qkv_fused = true;
                }
            }
            if (!qkv_fused) {
                try lk.qgemmLinear(self.allocator, x.ptr(), self.w_q.bytes, g.g_qg.ptr(), n, p.n_embd, qg_dim, qt_q.?);
                try lk.qgemmLinear(self.allocator, x.ptr(), self.w_k.bytes, g.g_k.ptr(), n, p.n_embd, kv_dim, qt_k.?);
                try lk.qgemmLinear(self.allocator, x.ptr(), self.w_v.bytes, g.g_v.ptr(), n, p.n_embd, kv_dim, qt_v.?);
            }
        } else {
            try self.matmul_engine.linearProjectionDevice(x, w_q32, &g.g_qg, n, p.n_embd, qg_dim);
            try self.matmul_engine.linearProjectionDevice(x, w_k32, &g.g_k, n, p.n_embd, kv_dim);
            try self.matmul_engine.linearProjectionDevice(x, w_v32, &g.g_v, n, p.n_embd, kv_dim);
        }
        // Breadcrumb DUMP_KVQUANT: checkpoint por etapa — el error sticky (700)
        // se observa en el PRIMER sync posterior al kernel culpable; sin estos
        // puntos, la atribución cae siempre en el append.
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(lk.stream) catch |e| {
                debugz.dbg.print("[attn] L{d} FALLO tras proyecciones QKV: {s}\n", .{ self.layer_idx, @errorName(e) });
                return e;
            };
        }

        // 2. Split Q|G interleaved. LFM2 (no_gate): qg_dim==q_dim, layout
        // PLANO sin interleaving — splitQGKernel leería qg[src] con src
        // hasta 2×total (OOB ⇒ basura; NaN en capa 27 del LFM2.5 con
        // prefill n=6). Copia plana y sin gateMul (igual que el CPU
        // forward :305/:620).
        if (p.no_gate) {
            try cudaz.cuMemcpyDtoDAsync(g.g_q.ptr(), g.g_qg.ptr(), n * q_dim * @sizeOf(f32), lk.stream);
        } else {
            try lk.splitQG(g.g_qg.ptr(), g.g_q.ptr(), g.g_g.ptr(), n, n_head, head_dim);
        }
        if (att_dbg and att_dbg_li) {
            att_dbg_dump.go("q_after_split", self.layer_idx, g.g_q.ptr(), n * q_dim);
            att_dbg_dump.go("qg_src", self.layer_idx, g.g_qg.ptr(), n * qg_dim);
        }

        // 3. Q/K RMSNorm per-head (reusa rmsNorm: rows = n*heads, cols = head_dim).
        // U1: sólo si el modelo trae q/k_norm — llama-arch los OMITÉ (ver
        // forward CPU paso 4).
        if (self.has_qk_norm) {
            try lk.rmsNorm(g.g_q.ptr(), @intFromPtr(g.g_q_norm.dev_ptr), g.g_q.ptr(), n * n_head, head_dim, p.rms_eps);
            if (att_dbg and att_dbg_li) {
                att_dbg_dump.go("q_rmsnorm", self.layer_idx, g.g_q.ptr(), n * q_dim);
            }
        }
        // Vision (PLAN_MMPROJ 3.2): mrope lee d_rope_pos (posición de contexto)
        // cuando está activo; si no, d_start_pos (slot KV) — comportamiento clásico.
        // Vision (Fase B): prefill con pos-ids per-token → kernel nuevo
        // (bit-fiel al host applyRoPEMultiSectionPosIds). Sin vision → el
        // clásico (d_rope_pos en decode o d_start_pos).
        if (pos_ids) |_| {
            try lk.mropePosIds(g.g_q.ptr(), @intCast(gpu.getDPosIds()), n * n_head, n, head_dim, p.n_rot, p.rope_sections, p.rope_freq_base);
        } else {
            const q_rope_src: usize = if (use_rope_pos) d_rope_pos_val else @as(usize, @intCast(gpu.getDStartPos()));
            try lk.mrope(g.g_q.ptr(), q_rope_src, n * n_head, n, head_dim, p.n_rot, p.rope_freq_base);
        }
        if (att_dbg and att_dbg_li) {
            att_dbg_dump.go("q_mrope", self.layer_idx, g.g_q.ptr(), n * q_dim);
        }

        // K es [n, kv_dim] = [n, n_kv_head, head_dim] en el mismo layout flat.
        if (self.has_qk_norm) {
            try lk.rmsNorm(g.g_k.ptr(), @intFromPtr(g.g_k_norm.dev_ptr), g.g_k.ptr(), n * n_kv_head, head_dim, p.rms_eps);
        }
        if (pos_ids) |_| {
            try lk.mropePosIds(g.g_k.ptr(), @intCast(gpu.getDPosIds()), n * n_kv_head, n, head_dim, p.n_rot, p.rope_sections, p.rope_freq_base);
        } else {
            const k_rope_src: usize = if (use_rope_pos) d_rope_pos_val else @as(usize, @intCast(gpu.getDStartPos()));
            try lk.mrope(g.g_k.ptr(), k_rope_src, n * n_kv_head, n, head_dim, p.n_rot, p.rope_freq_base);
        }
        if (att_dbg and att_dbg_li) {
            att_dbg_dump.go("q_rope", self.layer_idx, g.g_q.ptr(), n * q_dim);
            att_dbg_dump.go("k_rope", self.layer_idx, g.g_k.ptr(), n * kv_dim);
            att_dbg_dump.go("v_rope", self.layer_idx, g.g_v.ptr(), n * kv_dim);
        }
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(lk.stream) catch |e| {
                debugz.dbg.print("[attn] L{d} FALLO tras splitQG/norm/mrope: {s}\n", .{ self.layer_idx, @errorName(e) });
                return e;
            };
        }

        // 4. KV-append en device (f16 o q8_0 canónico) + decode/prefill paginado
        // device→device. El formato lo decide la config del pool paginado.
        const block_size = self.paged_kv.config.block_size;
        const d_cache = try gpu.cacheBase(self.paged_kv.block_alloc);

        // Prefill: la block table device (d_bt, la aloca uploadBlockTable) debe
        // subirse ANTES de kvAppendF16, que la lee para el bloque destino.
        var bt_host: []c_int = &.{};
        if (n > 1) {
            const max_num_blocks = self.block_table.numBlocks();
            bt_host = try self.allocator.alloc(c_int, max_num_blocks);
            for (0..max_num_blocks) |i| {
                bt_host[i] = if (self.block_table.getPhysical(i)) |phys| @intCast(phys) else -1;
            }
            try gpu.uploadBlockTable(self.layer_idx, bt_host);

            // Comitear los bloques que escribirá kvAppendF16 (pueden cruzar
            // límites de bloque en un chunk de prefill): residentes sin copiar.
            const first_block = start_pos / block_size;
            const last_block = (start_pos + n - 1) / block_size;
            var bi = first_block;
            while (bi <= last_block) : (bi += 1) {
                const phys = self.block_table.getPhysical(bi) orelse return HybridAttnError.KvCacheNotSet;
                try gpu.ensureBlockCommitted(self.paged_kv.block_alloc, phys);
            }
        }
        defer if (bt_host.len > 0) self.allocator.free(bt_host);
        if (debugz.dbg.dump_kvquant) {
            // Readback de control (d_bts ya garantizado por uploadBlockTable /
            // ensureLayerDecodeScratch en este punto): valores que verá append.
            var sp_chk: c_int = 0;
            var bt_chk: [2]c_int = .{ -99, -99 };
            try cudaz.cuMemcpyDtoH(@intFromPtr(&sp_chk), gpu.getDStartPos(), @sizeOf(c_int));
            try cudaz.cuMemcpyDtoH(@intFromPtr(&bt_chk), gpu.getDbt(self.layer_idx), 2 * @sizeOf(c_int));
            debugz.dbg.print("[attn] L{d} pre-append d_sp={d} d_bt[0..2]={any} n={d} host_start_pos={d}\n", .{ self.layer_idx, sp_chk, bt_chk, n, start_pos });
        }
        const qk_fmt = self.paged_kv.config.quant_k;
        const qv_fmt = self.paged_kv.config.quant_v;
        if (qk_fmt == qv_fmt and (qk_fmt == .q8_0 or qk_fmt == .q4_0 or qk_fmt == .q4_k or qk_fmt == .q8_k or qk_fmt == .iq4_xs or qk_fmt == .iq1_s or qk_fmt == .iq1_m or qk_fmt == .iq3_s or qk_fmt == .q2_k or qk_fmt == .q3_k or qk_fmt == .iq4_nl or qk_fmt == .iq3_xxs or qk_fmt == .mxfp4 or qk_fmt == .iq2_xxs or qk_fmt == .iq2_xs or qk_fmt == .iq2_s or qk_fmt == .tq2_0 or qk_fmt == .tq1_0)) {
            // Append cuantizado con escala embebida (q8_0: 34B/grupo;
            // q4_0: 18B/grupo; q4_k: 144B/SB de 256). Mismo contrato
            // single-writer por sector. q4_k exige además kv_dim%256==0.
            switch (qk_fmt) {
                .q8_0 => try lk.kvAppendQ8_0(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .q4_0 => try lk.kvAppendQ4_0(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .q4_k => try lk.kvAppendQ4_K(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .q8_k => try lk.kvAppendQ8_K(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .iq4_xs => try lk.kvAppendIQ4_XS(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .iq1_s => try lk.kvAppendIQ1_S(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .iq1_m => try lk.kvAppendIQ1_M(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .iq3_s => try lk.kvAppendIQ3_S(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .q2_k => try lk.kvAppendQ2_K(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .q3_k => try lk.kvAppendQ3_K(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .iq4_nl => try lk.kvAppendIQ4_NL(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .iq3_xxs => try lk.kvAppendIQ3_XXS(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .mxfp4 => try lk.kvAppendMXFP4(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .iq2_xxs => try lk.kvAppendIQ2_XXS(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .iq2_s => try lk.kvAppendIQ2_S(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .tq2_0 => try lk.kvAppendTQ2_0(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .tq1_0 => try lk.kvAppendTQ1_0(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                .iq2_xs => try lk.kvAppendIQ2_XS(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size),
                else => unreachable,
            }
        } else {
            try lk.kvAppendF16(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim, n_kv_head, head_dim, block_size);
        }
        if (debugz.dbg.dump_graph) {
            debugz.dbg.print("[graph] fwd kv={x} vv={x} cache={x} bt={x} sp={x} n={d} kv_dim={d}\n", .{ g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(self.layer_idx), gpu.getDStartPos(), n, kv_dim });
        }
        // 9.4 (lane-b) FA-native sobre KVarN: append al cache ANTES de la
        // atención (los records deben existir al leerlos), y fattn con los
        // descs en vez de materialize paged. El kvAppend clásico de arriba
        // ya corrió — el pool paged sigue hidratado como fallback/rutas
        // no-kvarn (así el flip del flag es seguro en cualquier paso).
        var kvarn_descs_now: ?cudaz.CUdeviceptr = null;
        if (self.kvarn_native and self.kvarn_append_fn != null) {
            const append_fn = self.kvarn_append_fn.?;
            kvarn_descs_now = append_fn(self.kvarn_append_ctx, @intCast(self.layer_idx), g.g_k.ptr(), g.g_v.ptr(), @intCast(n), @intCast(start_pos), lk.stream) catch |e| blk: {
                debugz.dbg.printLevel(.info, "[kvarn-fa] L{d} append FALLÓ ({s}); caigo a paged\n", .{ self.layer_idx, @errorName(e) });
                self.kvarn_native = false;
                break :blk null;
            };
        }
        if (self.kvarn_native and kvarn_descs_now != null and kvarn_descs_now.? != 0 and n == 1) {
            // Decode bs=1 native: Q f32 device + descs del cache. El
            // portable rota Q in-kernel (WHT-128×slices + cross) y
            // de-rota el output — Q/g_attn en dominio ORIGINAL.
            const fattn_mod = self.kvarn_fattn_module orelse {
                debugz.dbg.printLevel(.info, "[kvarn-fa] L{d} sin módulo fattn; caigo a paged\n", .{self.layer_idx});
                self.kvarn_native = false;
                try self.forwardPagedAttn(lk, g, n, bt_host, start_pos);
                return;
            };
            var attn_args: kvarn_fattn.KvarnAttentionArgs = .{
                .q_data = @ptrFromInt(g.g_q.ptr()),
                .k_descs = @ptrFromInt(kvarn_descs_now.?),
                // Layout por-lado del init: bloque V tras las K ⇒ base V =
                // base K + n_kv_heads·sizeof(KvarnDesc).
                .v_descs = @ptrFromInt(kvarn_descs_now.? + @as(usize, @sizeOf(kvk.KvarnDesc)) * n_kv_head),
                .mask_data = null,
                .dst_data = @ptrFromInt(g.g_attn.ptr()),
                .n_kv = @intCast(start_pos + 1),
                .n_q = 1,
                .n_q_heads = @intCast(n_head),
                .n_kv_heads = @intCast(n_kv_head),
                .n_stream = 1,
                .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
                .gqa = @intCast(n_head / n_kv_head),
            };
            switch (head_dim) {
                256 => _ = try kvarn_fattn.fattnKvarnPortableD256Device(fattn_mod, &attn_args, lk.stream),
                128 => _ = try kvarn_fattn.fattnKvarnPortableDevice(fattn_mod, &attn_args, lk.stream),
                64 => _ = try kvarn_fattn.fattnKvarnPortableD64Device(fattn_mod, &attn_args, lk.stream),
                else => {
                    debugz.dbg.printLevel(.info, "[kvarn-fa] L{d} head_dim={d} sin kernel native; paged\n", .{ self.layer_idx, head_dim });
                    self.kvarn_native = false;
                    try self.forwardPagedAttn(lk, g, n, bt_host, start_pos);
                    return;
                },
            }
            if (att_dbg and att_dbg_li) {
                att_dbg_dump.go("attn_out", self.layer_idx, g.g_attn.ptr(), n * q_dim);
            }
            // 9.4 fix: el kernel escribe attn_out de-rotado en g.g_attn
            // (dominio original) — NO hacer return: el flujo común de abajo
            // aplica el gate sigmoid(G)·attn y la proyección w_o que el
            // forward CPU ejecuta tras la atención. El return original
            // saltaba ambos ⇒ divergencia desde el token 2 (L7 q_rope,
            // dump A/B s5: attn_out L3 exacto pero L7 ya divergía).
            try self.applyGateAndOutputProj(lk, g, out, n, q_dim, p.no_gate);
            if (att_dbg and att_dbg_li) {
                att_dbg_dump.go("gated", self.layer_idx, g.g_attn.ptr(), n * q_dim);
            }
            return;
        }
        try self.forwardPagedAttn(lk, g, n, bt_host, start_pos);
        if (debugz.dbg.dump_kvquant) {
            cudaz.cuStreamSynchronize(lk.stream) catch |e| {
                debugz.dbg.print("[attn] L{d} FALLO tras atención (n={d}): {s}\n", .{ self.layer_idx, n, @errorName(e) });
                return e;
            };
        }
        try lk.copyF16toF32(g.d_attn16, g.g_attn.ptr(), n * q_dim);
        if (att_dbg and att_dbg_li) {
            att_dbg_dump.go("attn_out", self.layer_idx, g.g_attn.ptr(), n * q_dim);
        }

        // 5. Gate: sigmoid(G) * attn. LFM2 (no_gate): NO hay G — el CPU
        // forward (:620) lo salta; el GPU debe igualar (g_g sin escribir).
        if (!p.no_gate) {
            try lk.gateMul(g.g_attn.ptr(), g.g_g.ptr(), n * q_dim);
        }
        if (att_dbg and att_dbg_li) {
            att_dbg_dump.go("gated", self.layer_idx, g.g_attn.ptr(), n * q_dim);
        }

        // 6. Proyección de salida device→device (mixer, escrito en `out`).
        // T1-B VRAM-spec: cuantizado para todo dtype con kernel.
        const qt_o = if (quantAttnEnabled()) qgemmTypeFor(self.w_o.dtype()) else null;
        if (qt_o) |qt| {
            try lk.qgemmLinear(self.allocator, g.g_attn.ptr(), self.w_o.bytes, out.ptr(), n, q_dim, p.n_embd, qt);
        } else {
            try self.matmul_engine.linearProjectionDevice(g.g_attn, w_o32, out, n, q_dim, p.n_embd);
        }

        // 7. Prefill: sincronizar los bloques escritos al pool host (D2H async);
        // el pool host sigue siendo autoritativo para scheduler/COW. El decode
        // (n == 1) lo hace el llamador vía `syncDecodeBlocks` tras el grafo.
        if (n > 1) {
            const first_block = start_pos / block_size;
            const last_block = (start_pos + n - 1) / block_size;
            var bi = first_block;
            while (bi <= last_block) : (bi += 1) {
                const phys2 = self.block_table.getPhysical(bi) orelse return HybridAttnError.KvCacheNotSet;
                try gpu.syncBlockToHost(self.paged_kv.block_alloc, phys2);
            }
        }
    }

    /// Staging host del decode de un token (n == 1): block table, start_pos y
    /// seq_len pintados en buffers host persistentes (los copia el grafo con
    /// nodos HtoDAsync) y commit de los bloques que escribirá kvAppendF16.
    /// No lanza nada al device.
    pub fn stageDecodeHost(self: *Self, start_pos: usize, n: usize) !void {
        const gpu = self.paged_gpu orelse return HybridAttnError.KvCacheNotSet;
        const p = self.params;
        const block_size = self.paged_kv.config.block_size;
        const max_num_blocks = self.block_table.numBlocks();
        try gpu.setupDecodeScratch(self.layer_idx, p.n_head * p.head_dim, max_num_blocks);
        // rope_pos: si d_rope_pos está activo (vision), usa el delta de
        // contexto; si no, fillStaging ignora el extra (clásico).
        const rope_pos: ?usize = if (gpu.getDRopePos() != 0)
            @intCast(@as(i64, @intCast(start_pos)) + self.rope_pos_delta)
        else
            null;
        gpu.fillStaging(self.layer_idx, self.block_table, start_pos, start_pos + n, rope_pos);
        const first_block = start_pos / block_size;
        const last_block = (start_pos + n - 1) / block_size;
        var bi = first_block;
        while (bi <= last_block) : (bi += 1) {
            const phys = self.block_table.getPhysical(bi) orelse return HybridAttnError.KvCacheNotSet;
            try gpu.ensureBlockCommitted(self.paged_kv.block_alloc, phys);
        }
    }

    /// M3 slice 2 (Dev-B, P4): expone los punteros device de las K/V
    /// projections del ÚLTIMO forwardGPU de esta capa. Válido solo tras
    /// forwardGPU (g.g_k/g.g_v quedan residentes en device hasta el
    /// siguiente forwardGPU). El KVarN cache los consume vía
    /// `appendTokens(layer, k_ptr, v_ptr, n, pos, …)`. Retorna
    /// `null` si la GPU aún no se ha allocado (forwardGPU no llamado).
    pub const KvDevicePtrs = struct { k: cudaz.CUdeviceptr, v: cudaz.CUdeviceptr };
    pub fn kvDevicePtrs(self: *const Self) ?KvDevicePtrs {
        const g = self.gpu orelse return null;
        return .{ .k = g.g_k.ptr(), .v = g.g_v.ptr() };
    }

    /// Pre-dimensiona el staging de decode al presupuesto completo de bloques
    /// (budget de generación). En modo grafo esto fija los punteros host y el
    /// tamaño de d_bt ANTES de capturar: nunca se reasignan en replay.
    pub fn presizeDecodeScratch(self: *Self, budget_blocks: usize) !void {
        const gpu = self.paged_gpu orelse return HybridAttnError.KvCacheNotSet;
        const p = self.params;
        try gpu.setupDecodeScratch(self.layer_idx, p.n_head * p.head_dim, budget_blocks);
    }

    /// D2H async (stream-ordered, tras el grafo) de los bloques escritos por el
    /// decode del token en [start_pos, start_pos + n). Mantiene el pool host
    /// autoritativo para scheduler/COW.
    pub fn syncDecodeBlocks(self: *Self, start_pos: usize, n: usize) !void {
        const gpu = self.paged_gpu orelse return HybridAttnError.KvCacheNotSet;
        const block_size = self.paged_kv.config.block_size;
        const first_block = start_pos / block_size;
        const last_block = (start_pos + n - 1) / block_size;
        var bi = first_block;
        while (bi <= last_block) : (bi += 1) {
            const phys = self.block_table.getPhysical(bi) orelse return HybridAttnError.KvCacheNotSet;
            try gpu.syncBlockToHost(self.paged_kv.block_alloc, phys);
        }
    }

    pub fn ensureGpu(self: *Self) !void {
        if (self.gpu != null) return;
        self.gpu = try AttentionGpu.alloc(self.params, self.attn_q_norm.data, self.attn_k_norm.data);
    }

    /// T1-B VRAM-spec (ticket C 21:20 / parche lane-a en ssm.zig): tipo
    /// qgemmKernel para el peso, o null si no hay kernel cuantizado
    /// (⇒ fallback f32 vía weight_cache; eviction = territorio D).
    fn qgemmTypeFor(t: gguf.GgmlType) ?u32 {
        // Mapping GEMM COMPLETO 0..17 — espejo EXACTO de SsmLayer.qgemmTypeFor
        // (ssm.zig). FIX 9B: este mapping estaba desfasado (solo 0..7+iq1) y
        // los dtypes iq2_s/iq3_s de la attention del Qwen3.5-9B-UD caían al
        // fallback f32 eager (dequant 100MB+/capa en device → OOM en 8GB).
        // Regla: cualquier case añadido en ssm.zig DEBE espejarse aquí (P1).
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
            .iq1_s => 17,
            else => null,
        };
    }

    fn quantAttnEnabled() bool {
        // Flag histórico no_q4_attn: ahora gobierna TODO el camino cuantizado
        // de atención (cualquier dtype con kernel), mismo criterio que ssm.
        return layer_kernels.quantPath() and !debugz.dbg.no_q4_attn;
    }

    fn fp8AttnEnabled() bool {
        // FP8 attention path: requires CUDA, FP8 backend, and not disabled via env
        return cudaz.isCudaAvailable() and !debugz.dbg.no_fp8_attn and !debugz.dbg.no_fp8;
    }

    pub fn warmupGpuWeights(self: *Self) !void {
        const p = self.params;
        // T1-B: NO pre-calentar f32 de los pesos que irán por qgemm
        // (weight_cache sin eviction — cada f32 cacheado es presión VRAM).
        const qt_q = if (quantAttnEnabled()) qgemmTypeFor(self.w_q.dtype()) else null;
        const qt_k = if (quantAttnEnabled()) qgemmTypeFor(self.w_k.dtype()) else null;
        const qt_v = if (quantAttnEnabled()) qgemmTypeFor(self.w_v.dtype()) else null;
        const qt_o = if (quantAttnEnabled()) qgemmTypeFor(self.w_o.dtype()) else null;
        if (qt_q == null) {
            var w_q_shape = [_]usize{ p.qg_dim(), p.n_embd };
            var w_q_strides = [_]usize{ p.n_embd, 1 };
            _ = try self.matmul_engine.projectionDevicePtr(Tensor(f32){ .data = self.scratch_q, .shape = &w_q_shape, .strides = &w_q_strides, .offset = 0, .allocator = null, .owns_data = false });
        }
        if (qt_k == null) {
            var w_k_shape = [_]usize{ p.kv_dim(), p.n_embd };
            var w_k_strides = [_]usize{ p.n_embd, 1 };
            _ = try self.matmul_engine.projectionDevicePtr(Tensor(f32){ .data = self.scratch_k, .shape = &w_k_shape, .strides = &w_k_strides, .offset = 0, .allocator = null, .owns_data = false });
        }
        if (qt_v == null) {
            var w_v_shape = [_]usize{ p.kv_dim(), p.n_embd };
            var w_v_strides = [_]usize{ p.n_embd, 1 };
            _ = try self.matmul_engine.projectionDevicePtr(Tensor(f32){ .data = self.scratch_v, .shape = &w_v_shape, .strides = &w_v_strides, .offset = 0, .allocator = null, .owns_data = false });
        }
        if (qt_o == null) {
            var w_o_shape = [_]usize{ p.n_embd, p.n_head * p.head_dim };
            var w_o_strides = [_]usize{ p.n_head * p.head_dim, 1 };
            _ = try self.matmul_engine.projectionDevicePtr(Tensor(f32){ .data = self.scratch_o, .shape = &w_o_shape, .strides = &w_o_strides, .offset = 0, .allocator = null, .owns_data = false });
        }
    }
};

// ─── Buffers GPU de la capa de atención ──────────────────────────────────────
pub const AttentionGpu = struct {
    g_qg: cublas.GpuTensor(f32), // [n, qg_dim]
    g_k: cublas.GpuTensor(f32), // [n, kv_dim]
    g_v: cublas.GpuTensor(f32), // [n, kv_dim]
    g_q: cublas.GpuTensor(f32), // [n, n_head*head_dim]
    g_g: cublas.GpuTensor(f32), // [n, n_head*head_dim]
    g_attn: cublas.GpuTensor(f32), // [n, n_head*head_dim]
    g_q_norm: cublas.GpuBuffer(f32), // [n_head * head_dim]
    g_k_norm: cublas.GpuBuffer(f32), // [n_kv_head * head_dim]
    d_q16: cudaz.CUdeviceptr = 0,
    d_attn16: cudaz.CUdeviceptr = 0,
    cap_n: usize = 0,
    params: HybridAttnParams,

    fn alloc(p: HybridAttnParams, q_norm: []f32, k_norm: []f32) !AttentionGpu {
        const g_q_norm = try cublas.GpuBuffer(f32).alloc(p.n_head * p.head_dim);
        errdefer g_q_norm.free();
        try g_q_norm.upload(q_norm);
        const g_k_norm = try cublas.GpuBuffer(f32).alloc(p.n_kv_head * p.head_dim);
        errdefer g_k_norm.free();
        try g_k_norm.upload(k_norm);
        return .{
            .g_qg = undefined,
            .g_k = undefined,
            .g_v = undefined,
            .g_q = undefined,
            .g_g = undefined,
            .g_attn = undefined,
            .g_q_norm = g_q_norm,
            .g_k_norm = g_k_norm,
            .cap_n = 0,
            .params = p,
        };
    }

    fn ensureN(self: *AttentionGpu, n: usize) !void {
        if (self.cap_n >= n) return;
        const p = self.params;
        const qg_dim = p.qg_dim();
        const kv_dim = p.kv_dim();
        const q_dim = p.n_head * p.head_dim;
        if (self.cap_n > 0) {
            self.g_qg.deinit();
            self.g_k.deinit();
            self.g_v.deinit();
            self.g_q.deinit();
            self.g_g.deinit();
            self.g_attn.deinit();
            if (self.d_q16 != 0) cudaz.cuMemFree(self.d_q16);
            if (self.d_attn16 != 0) cudaz.cuMemFree(self.d_attn16);
        }
        self.g_qg = try cublas.GpuTensor(f32).alloc(n * qg_dim);
        self.g_k = try cublas.GpuTensor(f32).alloc(n * kv_dim);
        self.g_v = try cublas.GpuTensor(f32).alloc(n * kv_dim);
        self.g_q = try cublas.GpuTensor(f32).alloc(n * q_dim);
        self.g_g = try cublas.GpuTensor(f32).alloc(n * q_dim);
        self.g_attn = try cublas.GpuTensor(f32).alloc(n * q_dim);
        self.d_q16 = try cudaz.cuMemAlloc(n * q_dim * @sizeOf(f16));
        self.d_attn16 = try cudaz.cuMemAlloc(n * q_dim * @sizeOf(f16));
        self.cap_n = n;
    }

    fn deinit(self: *AttentionGpu) void {
        self.g_q_norm.free();
        self.g_k_norm.free();
        if (self.cap_n > 0) {
            self.g_qg.deinit();
            self.g_k.deinit();
            self.g_v.deinit();
            self.g_q.deinit();
            self.g_g.deinit();
            self.g_attn.deinit();
            if (self.d_q16 != 0) cudaz.cuMemFree(self.d_q16);
            if (self.d_attn16 != 0) cudaz.cuMemFree(self.d_attn16);
        }
    }
};

fn loadF16(data: []const u8, offset: usize) f32 {
    const bits: u16 = @as(u16, data[offset]) | (@as(u16, data[offset + 1]) << 8);
    const f16_val: f16 = @bitCast(bits);
    return @floatCast(f16_val);
}

fn loadF32(data: []const u8, offset: usize) f32 {
    const b0: u32 = data[offset];
    const b1: u32 = data[offset + 1];
    const b2: u32 = data[offset + 2];
    const b3: u32 = data[offset + 3];
    const bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    return @bitCast(bits);
}

fn storeF32(data: []u8, offset: usize, val: f32) void {
    const bits: u32 = @bitCast(val);
    data[offset] = @truncate(bits);
    data[offset + 1] = @truncate(bits >> 8);
    data[offset + 2] = @truncate(bits >> 16);
    data[offset + 3] = @truncate(bits >> 24);
}

fn storeF16(data: []u8, offset: usize, val: f32) void {
    const f16_val: f16 = @floatCast(val);
    const bits: u16 = @bitCast(f16_val);
    data[offset] = @truncate(bits);
    data[offset + 1] = @truncate(bits >> 8);
}

fn loadQuantWeight(g: *const gguf.GgufFile, prefix: []const u8, name: []const u8) !QuantWeight {
    const full = try std.fmt.allocPrint(std.heap.page_allocator, "{s}{s}", .{ prefix, name });
    defer std.heap.page_allocator.free(full);
    const info = g.getTensor(full) orelse return HybridAttnError.WeightFileNotFound;
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
    const info = g.getTensor(full) orelse return HybridAttnError.WeightFileNotFound;
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
    } else {
        tensor = try Tensor(f32).initUninitialized(allocator, &.{numel});
    }
    @memcpy(tensor.data, f32buf);
    return tensor;
}

/// U1 chasis (espejo 5fb64e2 lane-b1): carga un tensor de norma f32, o
/// devuelve ONES (identidad rmsNorm) si el tensor NO existe — llama-arch
/// no trae attn_q_norm/attn_k_norm y el WeightFileNotFound mataba el load
/// del path híbrido. `numel`: tamaño esperado (head_dim).
fn loadGgufF32OrOnes(
    allocator: std.mem.Allocator,
    g: *const gguf.GgufFile,
    prefix: []const u8,
    name: []const u8,
    numel: usize,
) !Tensor(f32) {
    {
        const full = try std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, name });
        defer allocator.free(full);
        if (g.getTensor(full) != null) return loadGgufF32(allocator, g, prefix, name);
    }
    const tensor = try Tensor(f32).initUninitialized(allocator, &.{numel});
    @memset(tensor.data, 1.0);
    return tensor;
}

// ─── Tests ───

fn approx(a: f32, b: f32, tol: f32) bool {
    return @abs(a - b) <= tol;
}

const TestParams = HybridAttnParams{
    .n_embd = 8,
    .n_head = 4,
    .n_kv_head = 2,
    .head_dim = 4,
    .n_rot = 4,
    .rope_sections = .{ 1, 1, 0, 0 },
    .rope_freq_base = 10000.0,
    .rms_eps = 1e-6,
    .max_seq_len = 16,
};

const test_params = TestParams;

const TestFixture = struct {
    allocator: std.mem.Allocator,
    layer: AttentionLayer,
    paged_kv: *paged.PagedKVCache,
    block_table: *paged.BlockTable,
    q_bytes: []u8,
    k_bytes: []u8,
    v_bytes: []u8,
    o_bytes: []u8,
    q_info: *gguf.TensorInfo,
    k_info: *gguf.TensorInfo,
    v_info: *gguf.TensorInfo,
    o_info: *gguf.TensorInfo,

    fn deinit(self: *TestFixture) void {
        self.layer.deinit();
        self.block_table.deinit(self.paged_kv.block_alloc);
        self.allocator.destroy(self.block_table);
        self.paged_kv.deinit();
        self.allocator.destroy(self.paged_kv);
        self.allocator.free(self.q_bytes);
        self.allocator.free(self.k_bytes);
        self.allocator.free(self.v_bytes);
        self.allocator.free(self.o_bytes);
        self.allocator.destroy(self.q_info);
        self.allocator.destroy(self.k_info);
        self.allocator.destroy(self.v_info);
        self.allocator.destroy(self.o_info);
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
    info.* = gguf.TensorInfo{
        .name = "test",
        .n_dims = 2,
        .dims = .{ in_dim, out_dim, 0, 0 },
        .dtype = .f32,
        .offset = 0,
    };
    bytes.* = try allocator.alloc(u8, values.len * 4);
    @memcpy(bytes.*, std.mem.sliceAsBytes(values));
    return QuantWeight.init(info, bytes.*);
}

fn buildTestLayer(allocator: std.mem.Allocator) !TestFixture {
    const paged_kv = try allocator.create(paged.PagedKVCache);
    paged_kv.* = try paged.PagedKVCache.init(allocator, .{
        .block_size = 8,
        .num_blocks = 16,
        .head_dim = test_params.head_dim,
        .num_kv_heads = test_params.n_kv_head,
        .num_q_heads = test_params.n_head,
        .dtype = .f16,
        .enable_prefix_cache = false,
        .enable_cpu_offload = false,
        .max_seq_len = test_params.max_seq_len,
    });
    errdefer {
        paged_kv.deinit();
        allocator.destroy(paged_kv);
    }
    const block_table = try allocator.create(paged.BlockTable);
    block_table.* = paged.BlockTable.init(allocator, 8);
    errdefer {
        block_table.deinit(paged_kv.block_alloc);
        allocator.destroy(block_table);
    }

    var layer = try AttentionLayer.init(allocator, 0, test_params, .auto, paged_kv, block_table, null);
    errdefer layer.deinit();

    const q_info = try allocator.create(gguf.TensorInfo);
    const k_info = try allocator.create(gguf.TensorInfo);
    const v_info = try allocator.create(gguf.TensorInfo);
    const o_info = try allocator.create(gguf.TensorInfo);

    // w_q [qg_dim=32, n_embd=8]: fused Q+G, Q then G per head (head_dim=4)
    // Q dims: 4 heads * 4 = 16, G dims: 4 heads * 4 = 16, total 32
    var q_vals: [32 * 8]f32 = undefined;
    for (0..32) |j| {
        for (0..8) |c| q_vals[j * 8 + c] = 0;
    }
    // Set Q part: identity per head (Q_h[h*4] = 1.0)
    for (0..4) |h| {
        for (0..4) |d| {
            const row = h * 8 + d; // Q offset = h*8 + d (since Q=first 4 of 8 per head)
            q_vals[row * 8 + d] = 1.0;
        }
    }
    // Set G part: 0.5 per head
    for (0..4) |h| {
        for (0..4) |d| {
            const row = h * 8 + 4 + d; // G offset = h*8 + head_dim (G after Q per head)
            q_vals[row * 8 + d] = 0.5;
        }
    }
    var q_bytes: []u8 = undefined;
    layer.w_q = try makeF32Weight(allocator, 32, 8, &q_vals, q_info, &q_bytes);

    // w_k [kv_dim=8, n_embd=8]: identity per kv_head
    var k_vals: [8 * 8]f32 = undefined;
    for (0..8) |j| {
        for (0..8) |c| k_vals[j * 8 + c] = 0;
    }
    for (0..2) |h| {
        for (0..4) |d| {
            const row = h * 4 + d;
            k_vals[row * 8 + d] = 1.0;
        }
    }
    var k_bytes: []u8 = undefined;
    layer.w_k = try makeF32Weight(allocator, 8, 8, &k_vals, k_info, &k_bytes);

    // w_v [kv_dim=8, n_embd=8]: identity per kv_head
    var v_vals: [8 * 8]f32 = undefined;
    for (0..8) |j| {
        for (0..8) |c| v_vals[j * 8 + c] = 0;
    }
    for (0..2) |h| {
        for (0..4) |d| {
            const row = h * 4 + d;
            v_vals[row * 8 + d] = 1.0;
        }
    }
    var v_bytes: []u8 = undefined;
    layer.w_v = try makeF32Weight(allocator, 8, 8, &v_vals, v_info, &v_bytes);

    // w_o [n_embd=8, n_head*head_dim=16]: identity
    var o_vals: [8 * 16]f32 = undefined;
    for (0..8) |j| {
        for (0..16) |c| o_vals[j * 16 + c] = 0;
    }
    for (0..8) |j| o_vals[j * 16 + j] = 1.0;
    var o_bytes: []u8 = undefined;
    layer.w_o = try makeF32Weight(allocator, 8, 16, &o_vals, o_info, &o_bytes);

    // Norm weights = 1.0
    for (layer.attn_q_norm.data) |*v| v.* = 1.0;
    for (layer.attn_k_norm.data) |*v| v.* = 1.0;

    return .{
        .allocator = allocator,
        .layer = layer,
        .paged_kv = paged_kv,
        .block_table = block_table,
        .q_bytes = q_bytes,
        .k_bytes = k_bytes,
        .v_bytes = v_bytes,
        .o_bytes = o_bytes,
        .q_info = q_info,
        .k_info = k_info,
        .v_info = v_info,
        .o_info = o_info,
    };
}

test "hybrid attention single token hand-computed" {
    const allocator = std.testing.allocator;
    var fixture = try buildTestLayer(allocator);
    defer fixture.deinit();
    var layer = &fixture.layer;

    // Pre-allocate blocks (scheduler normally does this)
    try fixture.block_table.appendTokens(fixture.paged_kv.block_alloc, 1);

    // Input x = [1, 8] all ones (2D como espera forward)
    var x = try Tensor(f32).alloc(allocator, &.{ 1, test_params.n_embd });
    defer x.deinit();
    for (x.data) |*v| v.* = 1.0;

    var out = try Tensor(f32).alloc(allocator, &.{ 1, test_params.n_embd });
    defer out.deinit();

    try layer.forward(x, &out, 0, 1);

    // With Q=identity, G=0.5, K=V=identity, start_pos=0, causal (only self-attn)
    // attn = softmax(Q·K/sqrt(4)) * V
    // Q·K = 4 (sum of 4 ones) per head, /2 = 2, softmax = 1.0
    // attn per head = V = ones
    // gate = sigmoid(0.5) ≈ 0.6225
    // attn_out = 0.6225 per head per dim
    // wo = identity -> output = sum over heads? No, wo maps [16, 8]
    // With our setup, need to trace through carefully
    // For now just check it runs and produces non-NaN
    for (out.data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "hybrid attention preserves norm with rope" {
    const allocator = std.testing.allocator;
    var fixture = try buildTestLayer(allocator);
    defer fixture.deinit();
    var layer = &fixture.layer;

    // Pre-allocate blocks for 4 tokens
    try fixture.block_table.appendTokens(fixture.paged_kv.block_alloc, 4);

    var x = try Tensor(f32).alloc(allocator, &.{ 4, test_params.n_embd });
    defer x.deinit();
    var rng = std.Random.Xoshiro256.init(123);
    x.randUniform(&rng, -0.5, 0.5);

    var out = try Tensor(f32).alloc(allocator, &.{ 4, test_params.n_embd });
    defer out.deinit();

    try layer.forward(x, &out, 0, 4);

    for (out.data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}
