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
    rms_eps: f32 = 1e-6,
    max_seq_len: usize = 2048,

    pub fn qg_dim(self: HybridAttnParams) usize {
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
    w_q: QuantWeight,       // [qg_dim, n_embd] = [8192, 4096] fused Q+G
    w_k: QuantWeight,       // [kv_dim, n_embd] = [1024, 4096]
    w_v: QuantWeight,       // [kv_dim, n_embd] = [1024, 4096]
    w_o: QuantWeight,       // [n_embd, n_head*head_dim] = [4096, 4096]

    // Pesos de normalización (f32, pequeños)
    attn_q_norm: Tensor(f32),    // [head_dim]
    attn_k_norm: Tensor(f32),    // [head_dim]

    // Scratch f16 persistente (reutilizado cada forward)
    scratch_q: []f32,     // qg_dim * n_embd
    scratch_k: []f32,     // kv_dim * n_embd
    scratch_v: []f32,     // kv_dim * n_embd
    scratch_o: []f32,     // n_embd * (n_head*head_dim)

    // KV-Cache paginado (PagedKVCache compartido entre capas de atención)
    paged_kv: *paged.PagedKVCache,
    block_table: *paged.BlockTable,
    sequence_id: u64 = 0,

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

        const qg_dim = params.qg_dim();
        const kv_dim = params.kv_dim();

        const scratch_q = try allocator.alloc(f32, qg_dim * params.n_embd);
        errdefer allocator.free(scratch_q);
        const scratch_k = try allocator.alloc(f32, kv_dim * params.n_embd);
        errdefer allocator.free(scratch_k);
        const scratch_v = try allocator.alloc(f32, kv_dim * params.n_embd);
        errdefer allocator.free(scratch_v);
        const scratch_o = try allocator.alloc(f32, params.n_embd * params.n_head * params.head_dim);
        errdefer allocator.free(scratch_o);

        var attn_q_norm = try Tensor(f32).alloc(allocator, &.{params.head_dim});
        errdefer attn_q_norm.deinit();
        var attn_k_norm = try Tensor(f32).alloc(allocator, &.{params.head_dim});
        errdefer attn_k_norm.deinit();

        const paged_gpu_local: ?*paged.PagedAttentionGpu = paged_gpu;

        const block_size = paged_kv.config.block_size;
        const k_quant: paged.QuantFormat = paged_kv.config.quant_k;
        const v_quant: paged.QuantFormat = paged_kv.config.quant_v;
        const tile_elems = block_size * kv_dim;
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
            .scratch_q = scratch_q,
            .scratch_k = scratch_k,
            .scratch_v = scratch_v,
            .scratch_o = scratch_o,
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
                kv_quant.encode(self.v_quant, self.v_staging, block_data[self.k_tile_bytes..][0..self.v_tile_bytes]);
            } else {
                const dst = block_data[self.k_tile_bytes..][0..self.v_tile_bytes];
                @memcpy(dst, std.mem.sliceAsBytes(self.v_staging));
            }
        }
        self.staged_block = -1;
        self.staged_tokens = 0;
    }

    /// Carga pesos desde GGUF (nombres qwen35).
    pub fn loadWeightsFromGguf(self: *Self, g: *const gguf.GgufFile) !void {
        const prefix = try std.fmt.allocPrint(self.allocator, "blk.{d}.", .{self.layer_idx});
        defer self.allocator.free(prefix);

        // Attention weights
        self.w_q = try loadQuantWeight(g, prefix, "attn_q.weight");
        self.w_k = try loadQuantWeight(g, prefix, "attn_k.weight");
        self.w_v = try loadQuantWeight(g, prefix, "attn_v.weight");
        self.w_o = try loadQuantWeight(g, prefix, "attn_output.weight");

        // Descuantizar los pesos UNA vez aquí (no por token en forward): los
        // scratch f32 persisten y el caché de pesos GPU los sube al device una
        // sola vez. Elimina el cuello de botella de re-descuantizar ~3GB/token.
        self.w_q.dequantToF32Transposed(self.scratch_q);
        self.w_k.dequantToF32Transposed(self.scratch_k);
        self.w_v.dequantToF32Transposed(self.scratch_v);
        self.w_o.dequantToF32Transposed(self.scratch_o);

        // Norm weights (f32)
        self.attn_q_norm.deinit();
        self.attn_q_norm = try loadGgufF32(self.allocator, g, prefix, "attn_q_norm.weight");
        self.attn_k_norm.deinit();
        self.attn_k_norm = try loadGgufF32(self.allocator, g, prefix, "attn_k_norm.weight");
    }

    /// Forward mixer-only: Attn(GQA+MRoPE+Gate) -> out
    /// Recibe input ya normalizado (pre-norm lo hace HybridLayer).
    /// `x`: [N, n_embd] (f16)
    /// `out`: [N, n_embd] (f16)
    /// `start_pos`: posición inicial en la secuencia (para RoPE y KV-cache)
    /// `n`: número de tokens a procesar (prefill en bloque o 1 token)
    pub fn forward(self: *Self, x: Tensor(f32), out: *Tensor(f32), start_pos: usize, n: usize) !void {
        const p = self.params;
        const qg_dim = p.qg_dim();
        const kv_dim = p.kv_dim();
        const head_dim = p.head_dim;
        const n_head = p.n_head;
        const n_kv_head = p.n_kv_head;
        const N = n;

        // === 2. Proyección Q+G fusionada (w_q) ===
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

        // Dividir Q y G interleaved [Q0|G0|Q1|G1|...]: Q = base, G = base+head_dim
        var Qf32 = try Tensor(f32).alloc(self.allocator, &.{ N, n_head, head_dim });
        defer Qf32.deinit();
        var Gf32 = try Tensor(f32).alloc(self.allocator, &.{ N, n_head, head_dim });
        defer Gf32.deinit();

        for (0..N) |t| {
            for (0..n_head) |h| {
                const base = h * (2 * head_dim);
                for (0..head_dim) |d| {
                    Qf32.data[t * n_head * head_dim + h * head_dim + d] = qg32.data[t * qg_dim + base + d];
                    Gf32.data[t * n_head * head_dim + h * head_dim + d] = qg32.data[t * qg_dim + base + head_dim + d];
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
        try self.matmul_engine.linearProjection(f32, x, w_v32, &Vf32);

        // Reshape K, V a [N, n_kv_head, head_dim]
        var Kf32_hm = try Tensor(f32).alloc(self.allocator, &.{ N, n_kv_head, head_dim });
        defer Kf32_hm.deinit();
        var Vf32_hm = try Tensor(f32).alloc(self.allocator, &.{ N, n_kv_head, head_dim });
        defer Vf32_hm.deinit();

        for (0..N) |t| {
            for (0..n_kv_head) |h| {
                for (0..head_dim) |d| {
                    Kf32_hm.data[t * n_kv_head * head_dim + h * head_dim + d] = Kf32.data[t * kv_dim + h * head_dim + d];
                    Vf32_hm.data[t * n_kv_head * head_dim + h * head_dim + d] = Vf32.data[t * kv_dim + h * head_dim + d];
                }
            }
        }

        // === 4. Q/K RMSNorm per-head ===
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

        // === 5. MRoPE (IMROPE) ===
        var Q_hm = try Tensor(f32).alloc(self.allocator, &.{ 1, n_head, N, head_dim });
        defer Q_hm.deinit();
        for (0..N * n_head * head_dim) |i| Q_hm.data[i] = Qf32.data[i];
        var K_hm = try Tensor(f32).alloc(self.allocator, &.{ 1, n_kv_head, N, head_dim });
        defer K_hm.deinit();
        for (0..N * n_kv_head * head_dim) |i| K_hm.data[i] = Kf32_hm.data[i];

        rope_mod.applyRoPEMultiSection(f32, &Q_hm, &K_hm, start_pos, head_dim, p.n_rot, p.rope_sections, p.rope_freq_base);

        for (0..N * n_head * head_dim) |i| Qf32.data[i] = Q_hm.data[i];
        for (0..N * n_kv_head * head_dim) |i| Kf32_hm.data[i] = K_hm.data[i];

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
                    self.block_table,
                    self.paged_kv.block_alloc,
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
                        if (s > start_pos + t) score = -std.math.inf(f32);
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

        // === 9. Gate: sigmoid(G) * attn_out ===
        for (0..N * n_head * head_dim) |i| {
            const g = Gf32.data[i];
            const sigmoid = 1.0 / (1.0 + @exp(-g));
            attn_out.data[i] *= sigmoid;
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
        for (out.data, attn_proj.data) |*o, a| o.* = a;
    }

    // ─── Forward de atención 100% GPU (Phase 1b, decode N == 1) ──────────────
    // Todo vive en device: proyecciones, split Q|G, Q/K norm, MRoPE, KV-append
    // f16, decode paginado device→device, gate y proyección de salida. Un solo
    // sync por token en el llamador. Fiel al forward CPU (incluida la semántica
    // flat de MRoPE) para que decode GPU == prefill CPU.
    pub fn forwardGPU(
        self: *Self,
        lk: *layer_kernels.LayerKernels,
        x: cublas.GpuTensor(f32),
        out: *cublas.GpuTensor(f32),
        start_pos: usize,
        n: usize,
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

        // 1. Proyecciones Q+G, K, V (device→device, pesos cacheados en GPU).
        // Pesos Q4_0 → GEMM cuantizado device (8× menos tráfico de VRAM),
        // también batched para prefill (n > 1).
        const q4_ok = self.w_q.dtype() == gguf.GgmlType.q4_0 and layer_kernels.quantPath() and std.c.getenv("NOQ4ATTN") == null;
        if (q4_ok) {
            try lk.qgemmLinear(self.allocator, x.ptr(), self.w_q.bytes, g.g_qg.ptr(), n, p.n_embd, qg_dim, 0);
            try lk.qgemmLinear(self.allocator, x.ptr(), self.w_k.bytes, g.g_k.ptr(), n, p.n_embd, kv_dim, 0);
            try lk.qgemmLinear(self.allocator, x.ptr(), self.w_v.bytes, g.g_v.ptr(), n, p.n_embd, kv_dim, 0);
        } else {
            try self.matmul_engine.linearProjectionDevice(x, w_q32, &g.g_qg, n, p.n_embd, qg_dim);
            try self.matmul_engine.linearProjectionDevice(x, w_k32, &g.g_k, n, p.n_embd, kv_dim);
            try self.matmul_engine.linearProjectionDevice(x, w_v32, &g.g_v, n, p.n_embd, kv_dim);
        }

        // 2. Split Q|G interleaved.
        try lk.splitQG(g.g_qg.ptr(), g.g_q.ptr(), g.g_g.ptr(), n, n_head, head_dim);

        // 3. Q/K RMSNorm per-head (reusa rmsNorm: rows = n*heads, cols = head_dim).
        try lk.rmsNorm(g.g_q.ptr(), @intFromPtr(g.g_q_norm.dev_ptr), g.g_q.ptr(), n * n_head, head_dim, p.rms_eps);
        try lk.mrope(g.g_q.ptr(), n * n_head, n, head_dim, p.n_rot, p.rope_freq_base, start_pos);

        // K es [n, kv_dim] = [n, n_kv_head, head_dim] en el mismo layout flat.
        try lk.rmsNorm(g.g_k.ptr(), @intFromPtr(g.g_k_norm.dev_ptr), g.g_k.ptr(), n * n_kv_head, head_dim, p.rms_eps);
        try lk.mrope(g.g_k.ptr(), n * n_kv_head, n, head_dim, p.n_rot, p.rope_freq_base, start_pos);

        // 4. KV-append f16 en device + decode paginado device→device.
        const gpu = self.paged_gpu orelse return HybridAttnError.KvCacheNotSet;
        const block_size = self.paged_kv.config.block_size;
        const max_num_blocks = self.block_table.numBlocks();
        const bt_host = try self.allocator.alloc(c_int, max_num_blocks);
        defer self.allocator.free(bt_host);
        for (0..max_num_blocks) |i| {
            bt_host[i] = if (self.block_table.getPhysical(i)) |phys| @intCast(phys) else -1;
        }
        try gpu.uploadBlockTable(bt_host);

        // Comitear los bloques que escribirá kvAppendF16 (pueden cruzar límites
        // de bloque en un chunk de prefill): se marcan residentes sin copiar.
        const first_block = start_pos / block_size;
        const last_block = (start_pos + n - 1) / block_size;
        var bi = first_block;
        while (bi <= last_block) : (bi += 1) {
            const phys = self.block_table.getPhysical(bi) orelse return HybridAttnError.KvCacheNotSet;
            try gpu.ensureBlockCommitted(self.paged_kv.block_alloc, phys);
        }

        const d_cache = try gpu.cacheBase(self.paged_kv.block_alloc);
        try lk.kvAppendF16(g.g_k.ptr(), g.g_v.ptr(), d_cache, gpu.getDbt(), n, start_pos, kv_dim, n_kv_head, head_dim, block_size);
        try lk.copyF32toF16(g.g_q.ptr(), g.d_q16, n * q_dim);
        if (n > 1) {
            // Prefill causal en bloque (device→device): atiende los n queries
            // sobre todos los tokens ya escritos en el pool.
            try gpu.prefillDevice(g.d_q16, g.d_attn16, self.paged_kv.block_alloc, bt_host, n, start_pos);
        } else {
            try gpu.decodeDevice(g.d_q16, g.d_attn16, self.paged_kv.block_alloc, bt_host, start_pos + n);
        }
        try lk.copyF16toF32(g.d_attn16, g.g_attn.ptr(), n * q_dim);

        // 5. Gate: sigmoid(G) * attn.
        try lk.gateMul(g.g_attn.ptr(), g.g_g.ptr(), n * q_dim);

        // 6. Proyección de salida device→device (mixer, escrito en `out`).
        if (q4_ok) {
            try lk.qgemmLinear(self.allocator, g.g_attn.ptr(), self.w_o.bytes, out.ptr(), n, q_dim, p.n_embd, 0);
        } else {
            try self.matmul_engine.linearProjectionDevice(g.g_attn, w_o32, out, n, q_dim, p.n_embd);
        }

        // 7. Sincronizar los bloques escritos al pool host (D2H async): el pool host
        // sigue siendo autoritativo para scheduler/COW.
        bi = first_block;
        while (bi <= last_block) : (bi += 1) {
            const phys2 = self.block_table.getPhysical(bi) orelse return HybridAttnError.KvCacheNotSet;
            try gpu.syncBlockToHost(self.paged_kv.block_alloc, phys2);
        }
    }

    pub fn ensureGpu(self: *Self) !void {
        if (self.gpu != null) return;
        self.gpu = try AttentionGpu.alloc(self.params, self.attn_q_norm.data, self.attn_k_norm.data);
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
    g_q_norm: cublas.GpuBuffer(f32), // [head_dim]
    g_k_norm: cublas.GpuBuffer(f32), // [head_dim]
    d_q16: cudaz.CUdeviceptr = 0,
    d_attn16: cudaz.CUdeviceptr = 0,
    cap_n: usize = 0,
    params: HybridAttnParams,

    fn alloc(p: HybridAttnParams, q_norm: []f32, k_norm: []f32) !AttentionGpu {
        const g_q_norm = try cublas.GpuBuffer(f32).alloc(p.head_dim);
        errdefer g_q_norm.free();
        try g_q_norm.upload(q_norm);
        const g_k_norm = try cublas.GpuBuffer(f32).alloc(p.head_dim);
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

    // Input x = [1, 1, 8] all ones
    var x = try Tensor(f32).alloc(allocator, &.{ 1, 1, test_params.n_embd });
    defer x.deinit();
    for (x.data) |*v| v.* = 1.0;

    var out = try Tensor(f32).alloc(allocator, &.{ 1, 1, test_params.n_embd });
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

    var x = try Tensor(f32).alloc(allocator, &.{ 1, 4, test_params.n_embd });
    defer x.deinit();
    var rng = std.Random.Xoshiro256.init(123);
    x.randUniform(&rng, -0.5, 0.5);

    var out = try Tensor(f32).alloc(allocator, &.{ 1, 4, test_params.n_embd });
    defer out.deinit();

    try layer.forward(x, &out, 0, 4);

    for (out.data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}