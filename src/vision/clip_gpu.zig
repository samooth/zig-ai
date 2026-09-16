//! clip_gpu — encoder ViT device-resident (PLAN_MMPROJ 10.2, lane-mmproj).
//!
//! vs el path CPU (clip_block/clip_attention): las ACTIVACIONES viven en
//! GPU todo el encode (subida única de la imagen, bajada única de los
//! embeddings), pesos residentes por bloque (subida única lazy). Los GEMMs
//! son cuBLAS device→device (gemmF32DeviceResident); LN/RMS/GELU/bias/
//! atención/MRoPE via kernels del .cu. Objetivo: 44s → segundos.
//!
//! Paridad certificable: golden test (tests/test_mmproj.zig e2e) — este
//! path NO cambia los pesos (los mismos del ClipBlock host) y el gate
//! MMPROJ_GPU=0 fuerza el path CPU para A/B.
const std = @import("std");
const Tensor = @import("core").Tensor;
const cublas = @import("cublas");
const cudaz = @import("cudaz");
const layer_kernels = @import("layer_kernels");
const mrope = @import("mrope_vision");
const debugz = @import("debug");

pub const ClipGpuError = error{
    CudaUnavailable,
    OutOfMemory,
    ShapeMismatch,
};

pub const GpuBuffer = cublas.GpuBuffer(f32);

/// Pesos de un bloque del ViT residentes en GPU. Subida única.
pub const GpuBlockWeights = struct {
    // attn
    qkv_w: GpuBuffer, // [3·n_embd, n_embd] W_T row-major (transposed host)
    qkv_b: GpuBuffer, // [3·n_embd]
    o_w: GpuBuffer, // [n_embd, n_embd]
    o_b: GpuBuffer, // [n_embd]
    // ln
    ln1_w: GpuBuffer,
    ln1_b: GpuBuffer, // [n_embd] (zeros si RMS)
    ln2_w: GpuBuffer,
    ln2_b: GpuBuffer,
    // ffn
    ff_up: GpuBuffer, // [n_ff, n_embd]
    ff_up_b: GpuBuffer, // [n_ff]
    ff_down: GpuBuffer, // [n_embd, n_ff]
    ff_down_b: GpuBuffer, // [n_embd]

};

/// Pesos deepstack de un bloque (Qwen3-VL). Subida única.
pub const GpuDeepstackWeights = struct {
    norm_w: GpuBuffer,
    norm_b: GpuBuffer, // [ds_in] (zeros si RMS)
    fc1_w: GpuBuffer, // [n_ff, ds_in] W_T row-major
    fc1_b: GpuBuffer, // [n_ff]
    fc2_w: GpuBuffer, // [ds_out_dim, n_ff] W_T row-major
    fc2_b: GpuBuffer, // [ds_out_dim]
    n_ff: usize,
    ds_in: usize, // n_embd * merge² (4)
    ds_out_dim: usize,
};

/// Pesos deepstack en HOST (dequantizados, referenciados por el ClipBlock).
pub const DeepstackHostWeights = struct {
    norm_w: []const f32,
    norm_b: []const f32,
    fc1_w: []const f32, // [n_ff, n_embd*4]
    fc1_b: []const f32,
    fc2_w: []const f32, // [ds_out_dim, n_ff]
    fc2_b: []const f32,
    n_ff: usize,
    ds_in: usize,
    ds_out_dim: usize,
    /// Índice del bloque en `blocks` al que pertenece este deepstack.
    block_idx: usize,
};

/// Estado GPU del encoder completo: pesos por bloque + buffers de activación
/// persistentes (dimensionados al n_pos máx. de la imagen).
pub const GpuClipEncoder = struct {
    allocator: std.mem.Allocator,
    handle: cublas.CuBlasHandle,
    lk: *layer_kernels.LayerKernels,
    blocks: []GpuBlockWeights,
    n_embd: usize,
    n_ff: usize,
    n_head: usize,
    head_dim: usize,
    n_pos: usize,
    use_rms_norm: bool,
    eps: f32,

    // Buffers de activación persistentes (device)
    d_x: GpuBuffer, // input del bloque [n_pos, n_embd] (residual)
    d_ln1: GpuBuffer, // tras LN1
    d_qkv: GpuBuffer, // tras QKV proj [n_pos, 3·n_embd]
    d_attn_out: GpuBuffer, // tras attention [n_pos, n_embd]
    d_inp_l: GpuBuffer, // residual 1
    d_ln2: GpuBuffer,
    d_mid: GpuBuffer, // FFN mid [n_pos, n_ff]
    d_ffn: GpuBuffer, // [n_pos, n_embd]
    // packs por head para la atención (device)
    d_pack_q: GpuBuffer, // [n_head][n_pos·hd]
    d_pack_k: GpuBuffer,
    d_pack_v: GpuBuffer,
    d_pack_o: GpuBuffer,
    // MRoPE staging device [n_pos·4] i32
    d_pos_ids: cudaz.CUdeviceptr = 0,
    // pos_ids host staging (pinned no necesario; upload por imagen)
    pos_ids_host: [][4]i32,

    // Deepstack (Qwen3-VL merge2=4)
    has_deepstack: bool = false,
    ds: []GpuDeepstackWeights = &[_]GpuDeepstackWeights{},
    ds_block_idx: usize = 0,
    d_ds_out: GpuBuffer = undefined,
    d_ds_view: GpuBuffer = undefined,

    const Self = @This();

    /// Construye el estado GPU desde los pesos HOST ya dequantizados del
    /// ClipEncoder (clip_block fields). `blocks_host`: slices de pesos por
    /// bloque en el MISMO orden que clip_encoder.blocks.
    pub fn init(
        allocator: std.mem.Allocator,
        lk: *layer_kernels.LayerKernels,
        blocks_host: []const BlockHostWeights,
        cfg_n_embd: usize,
        cfg_n_ff: usize,
        cfg_n_head: usize,
        cfg_head_dim: usize,
        cfg_n_pos: usize,
        cfg_use_rms_norm: bool,
        cfg_eps: f32,
        ds_host: []const DeepstackHostWeights,
    ) !Self {
        try cudaz.ensureContext();
        const handle = try cublas.CuBlasHandle.init();

        const blocks = try allocator.alloc(GpuBlockWeights, blocks_host.len);
        errdefer allocator.free(blocks);
        for (blocks_host, 0..) |bh, i| {
            blocks[i] = try uploadBlock(allocator, bh, cfg_n_embd);
        }

        // Deepstack weights (si hay)
        const has_ds = ds_host.len > 0;
        const empty_ds: [0]GpuDeepstackWeights = .{};
        const ds = if (has_ds) blk: {
            const ds_arr = try allocator.alloc(GpuDeepstackWeights, ds_host.len);
            for (ds_host, 0..) |dsh, i| {
                ds_arr[i] = .{
                    .norm_w = try upBuf(dsh.norm_w),
                    .norm_b = if (dsh.norm_b.len > 0) try upBuf(dsh.norm_b) else try upZeros(allocator, dsh.ds_in),
                    .fc1_w = try upBuf(dsh.fc1_w),
                    .fc1_b = try upBuf(dsh.fc1_b),
                    .fc2_w = try upBuf(dsh.fc2_w),
                    .fc2_b = try upBuf(dsh.fc2_b),
                    .n_ff = dsh.n_ff,
                    .ds_in = dsh.ds_in,
                    .ds_out_dim = dsh.ds_out_dim,
                };
            }
            break :blk ds_arr;
        } else @constCast(empty_ds[0..]);

        const A = cfg_n_pos * cfg_n_embd;
        const H = cfg_n_head * cfg_n_pos * cfg_head_dim;

        // Deepstack buffers
        const merge2: usize = 4;
        const ds_in_dim = cfg_n_embd * merge2;
        const ds_view_len = cfg_n_pos * ds_in_dim; // [n_pos, n_embd*4]
        const ds_out_len = if (has_ds) ds_host[0].ds_out_dim * cfg_n_pos else 0;

        const s: Self = .{
            .allocator = allocator,
            .handle = handle,
            .lk = lk,
            .blocks = blocks,
            .n_embd = cfg_n_embd,
            .n_ff = cfg_n_ff,
            .n_head = cfg_n_head,
            .head_dim = cfg_head_dim,
            .n_pos = cfg_n_pos,
            .use_rms_norm = cfg_use_rms_norm,
            .eps = cfg_eps,
            .d_x = try GpuBuffer.alloc(A),
            .d_ln1 = try GpuBuffer.alloc(A),
            .d_qkv = try GpuBuffer.alloc(3 * A),
            .d_attn_out = try GpuBuffer.alloc(A),
            .d_inp_l = try GpuBuffer.alloc(A),
            .d_ln2 = try GpuBuffer.alloc(A),
            .d_mid = try GpuBuffer.alloc(cfg_n_pos * cfg_n_ff),
            .d_ffn = try GpuBuffer.alloc(A),
            .d_pack_q = try GpuBuffer.alloc(H),
            .d_pack_k = try GpuBuffer.alloc(H),
            .d_pack_v = try GpuBuffer.alloc(H),
            .d_pack_o = try GpuBuffer.alloc(H),
            .d_ds_out = if (has_ds) try GpuBuffer.alloc(ds_out_len) else undefined,
            .d_ds_view = if (has_ds) try GpuBuffer.alloc(ds_view_len) else undefined,
            .has_deepstack = has_ds,
            .ds = ds,
            .ds_block_idx = if (has_ds) ds_host[0].block_idx else 0,
            .pos_ids_host = try allocator.alloc([4]i32, cfg_n_pos),
        };
        return s;
    }

    pub fn deinit(self: *Self) void {
        for (self.blocks) |*b| freeBlock(b);
        self.allocator.free(self.blocks);
        self.d_x.free();
        self.d_ln1.free();
        self.d_qkv.free();
        self.d_attn_out.free();
        self.d_inp_l.free();
        self.d_ln2.free();
        self.d_mid.free();
        self.d_ffn.free();
        self.d_pack_q.free();
        self.d_pack_k.free();
        self.d_pack_v.free();
        self.d_pack_o.free();
        self.allocator.free(self.pos_ids_host);
        if (self.d_pos_ids != 0) cudaz.cuMemFree(self.d_pos_ids);
        if (self.has_deepstack) {
            for (self.ds) |*d| {
                inline for (@typeInfo(GpuDeepstackWeights).@"struct".fields) |f| {
                    if (f.type != usize) @field(d, f.name).free();
                }
            }
            self.allocator.free(self.ds);
            self.d_ds_out.free();
            self.d_ds_view.free();
        }
    }

    /// Multi-imagen: redimensiona los buffers de activación si la nueva
    /// imagen necesita más n_pos que la actual (grow-only; los PESOS de los
    /// bloques nunca se re-suben). Con n_pos <= actual es no-op: imágenes
    /// consecutivas del mismo tamaño pagan sólo los launches.
    pub fn ensureN(self: *Self, want_n_pos: usize) !void {
        if (want_n_pos <= self.n_pos) return;
        const A_old = self.n_pos * self.n_embd;
        const H_old = self.n_head * self.n_pos * self.head_dim;
        _ = A_old;
        _ = H_old;
        self.d_x.free();
        self.d_ln1.free();
        self.d_qkv.free();
        self.d_attn_out.free();
        self.d_inp_l.free();
        self.d_ln2.free();
        self.d_mid.free();
        self.d_ffn.free();
        self.d_pack_q.free();
        self.d_pack_k.free();
        self.d_pack_v.free();
        self.d_pack_o.free();
        if (self.d_pos_ids != 0) {
            cudaz.cuMemFree(self.d_pos_ids);
            self.d_pos_ids = 0;
        }
        self.allocator.free(self.pos_ids_host);

        const A = want_n_pos * self.n_embd;
        const H = self.n_head * want_n_pos * self.head_dim;
        self.n_pos = want_n_pos;
        self.d_x = try GpuBuffer.alloc(A);
        self.d_ln1 = try GpuBuffer.alloc(A);
        self.d_qkv = try GpuBuffer.alloc(3 * A);
        self.d_attn_out = try GpuBuffer.alloc(A);
        self.d_inp_l = try GpuBuffer.alloc(A);
        self.d_ln2 = try GpuBuffer.alloc(A);
        self.d_mid = try GpuBuffer.alloc(want_n_pos * self.n_ff);
        self.d_ffn = try GpuBuffer.alloc(A);
        self.d_pack_q = try GpuBuffer.alloc(H);
        self.d_pack_k = try GpuBuffer.alloc(H);
        self.d_pack_v = try GpuBuffer.alloc(H);
        self.d_pack_o = try GpuBuffer.alloc(H);
        self.pos_ids_host = try self.allocator.alloc([4]i32, want_n_pos);
        if (self.has_deepstack) {
            self.d_ds_view.free();
            self.d_ds_out.free();
            const merge2: usize = 4;
            const ds_view_len = want_n_pos * self.n_embd * merge2;
            const ds_out_len = self.ds[0].ds_out_dim * want_n_pos;
            self.d_ds_view = try GpuBuffer.alloc(ds_view_len);
            self.d_ds_out = try GpuBuffer.alloc(ds_out_len);
        }
    }

    fn freeBlock(b: *GpuBlockWeights) void {
        inline for (@typeInfo(GpuBlockWeights).@"struct".fields) |f| {
            @field(b, f.name).free();
        }
    }

    /// Pesos host de UN bloque (los del ClipBlock CPU ya dequantizados).
    pub const BlockHostWeights = struct {
        qkv_w: []const f32, // [3·n_embd, n_embd] (qkv_w_t.data)
        qkv_b: []const f32,
        o_w: []const f32,
        o_b: []const f32,
        ln1_w: []const f32,
        ln1_b: []const f32,
        ln2_w: []const f32,
        ln2_b: []const f32,
        ff_up: []const f32, // [n_ff, n_embd]
        ff_up_b: []const f32,
        ff_down: []const f32, // [n_embd, n_ff]
        ff_down_b: []const f32,
    };

    fn upBuf(bytes: []const f32) !GpuBuffer {
        var g = try GpuBuffer.alloc(bytes.len);
        errdefer g.free();
        try g.upload(bytes);
        return g;
    }

    fn upZeros(allocator: std.mem.Allocator, n: usize) !GpuBuffer {
        const z = try allocator.alloc(f32, n);
        defer allocator.free(z);
        @memset(z, 0);
        return upBuf(z);
    }

    fn uploadBlock(allocator: std.mem.Allocator, bh: BlockHostWeights, n_embd: usize) !GpuBlockWeights {
        return .{
            .qkv_w = try upBuf(bh.qkv_w),
            .qkv_b = try upBuf(bh.qkv_b),
            .o_w = try upBuf(bh.o_w),
            .o_b = try upBuf(bh.o_b),
            .ln1_w = try upBuf(bh.ln1_w),
            .ln1_b = if (bh.ln1_b.len > 0) try upBuf(bh.ln1_b) else try upZeros(allocator, n_embd),
            .ln2_w = try upBuf(bh.ln2_w),
            .ln2_b = if (bh.ln2_b.len > 0) try upBuf(bh.ln2_b) else try upZeros(allocator, n_embd),
            .ff_up = try upBuf(bh.ff_up),
            .ff_up_b = try upBuf(bh.ff_up_b),
            .ff_down = try upBuf(bh.ff_down),
            .ff_down_b = try upBuf(bh.ff_down_b),
        };
    }

    /// Ejecuta UN bloque del ViT en GPU: d_x (in/residual) → d_ffn (out).
    /// d_ffn queda listo como input del siguiente bloque (caller hace swap
    /// o copia device→device).
    pub fn forwardBlock(self: *Self, il: usize) !void {
        const b = &self.blocks[il];
        const n = self.n_pos;
        const ne = self.n_embd;
        const A = n * ne;

        // ── 1. LN1/RMS1
        if (debugz.dbg.at(.info)) debugz.dbg.printLevel(.info, "[gpu-vit] blk {d}: LN1\n", .{il});
        if (self.use_rms_norm) {
            try self.lk.rmsNorm(@intFromPtr(self.d_x.dev_ptr), @intFromPtr(b.ln1_w.dev_ptr), @intFromPtr(self.d_ln1.dev_ptr), n, ne, self.eps);
        } else {
            try self.lk.layerNormDev(@intFromPtr(self.d_x.dev_ptr), @intFromPtr(b.ln1_w.dev_ptr), @intFromPtr(b.ln1_b.dev_ptr), @intFromPtr(self.d_ln1.dev_ptr), n, ne, self.eps);
        }

        if (debugz.dbg.at(.info)) debugz.dbg.printLevel(.info, "[gpu-vit] blk {d}: QKV\n", .{il});
        // ── 2. QKV: d_qkv = ln1 · qkv_wᵀ + bias (cuBLAS D2D)
        try cublas.gemmF32DeviceResident(
            self.handle,
            self.d_ln1.dev_ptr,
            b.qkv_w.dev_ptr,
            self.d_qkv.dev_ptr,
            n,
            3 * ne,
            ne,
            false,
            true,
        );
        try self.lk.biasAddDev(@intFromPtr(self.d_qkv.dev_ptr), @intFromPtr(b.qkv_b.dev_ptr), 3 * A, 3 * ne);

        if (debugz.dbg.at(.info)) debugz.dbg.printLevel(.info, "[gpu-vit] blk {d}: PACK\n", .{il});
        // ── 3. Pack por head + MRoPE device
        // QKV row-interleaved [n][q|k|v cada ne] — el pack lee el slice del
        // bloque con base = offset DENTRO de la fila (0, ne, 2·ne elems) y
        // stride 3·ne. 10.2-bisect: antes las bases usaban offsets de
        // bloque contiguo (seg_dev = n·ne) y el kernel asumía stride ne
        // ⇒ K/V pack leían filas de Q.
        const elem_dev = @sizeOf(f32);
        const row_stride: usize = 3 * ne;
        const qkv_base: usize = @intFromPtr(self.d_qkv.dev_ptr);
        const pack_q_base: usize = @intFromPtr(self.d_pack_q.dev_ptr);
        const pack_k_base: usize = @intFromPtr(self.d_pack_k.dev_ptr);
        const pack_v_base: usize = @intFromPtr(self.d_pack_v.dev_ptr);
        const pack_o_base: usize = @intFromPtr(self.d_pack_o.dev_ptr);
        try self.lk.packHead(qkv_base, pack_q_base, n, self.n_head, self.head_dim, row_stride);
        try self.lk.packHead(qkv_base + ne * elem_dev, pack_k_base, n, self.n_head, self.head_dim, row_stride);
        try self.lk.packHead(qkv_base + 2 * ne * elem_dev, pack_v_base, n, self.n_head, self.head_dim, row_stride);

        if (self.d_pos_ids == 0) {
            self.d_pos_ids = try cudaz.cuMemAlloc(n * 4 * @sizeOf(i32));
        }
        try cudaz.cuMemcpyHtoD(self.d_pos_ids, @intFromPtr(self.pos_ids_host.ptr), n * 4 * @sizeOf(i32));
        const hd_pack: usize = self.head_dim;
        const sections = [4]usize{ self.head_dim / 4, self.head_dim / 4, self.head_dim / 4, self.head_dim / 4 };
        const pack_slice: usize = @sizeOf(f32) * n * hd_pack;
        for (0..self.n_head) |h| {
            // 10.2-bisect: el ViT usa M-RoPE VISION interleaved (pares
            // adyacentes, freq por n_pairs) — NO el NEOX half-split del
            // mropePosIdsKernel del LLM. Con ids 2D de imagen divergían.
            try self.lk.mropeVision(pack_q_base + h * pack_slice, self.d_pos_ids, n, n, hd_pack, sections);
            try self.lk.mropeVision(pack_k_base + h * pack_slice, self.d_pos_ids, n, n, hd_pack, sections);
        }

        if (debugz.dbg.at(.info)) debugz.dbg.printLevel(.info, "[gpu-vit] blk {d}: ROPE+ATTN\n", .{il});
        // ── 4. Atención por head (bidireccional) + unpack + O proj + bias
        for (0..self.n_head) |h| {
            try self.lk.vitAttnHead(pack_q_base + h * pack_slice, pack_k_base + h * pack_slice, pack_v_base + h * pack_slice, pack_o_base + h * pack_slice, n, hd_pack);
        }
        try self.lk.unpackHead(pack_o_base, @intFromPtr(self.d_attn_out.dev_ptr), n, self.n_head, hd_pack);
        try cublas.gemmF32DeviceResident(self.handle, self.d_attn_out.dev_ptr, b.o_w.dev_ptr, self.d_ffn.dev_ptr, n, ne, ne, false, true);
        try self.lk.biasAddDev(@intFromPtr(self.d_ffn.dev_ptr), @intFromPtr(b.o_b.dev_ptr), A, ne);

        if (debugz.dbg.at(.info)) debugz.dbg.printLevel(.info, "[gpu-vit] blk {d}: RES1\n", .{il});
        // ── 5. Residual 1: inp_l = x + attn_out
        try self.lk.add(@intFromPtr(self.d_x.dev_ptr), @intFromPtr(self.d_ffn.dev_ptr), @intFromPtr(self.d_inp_l.dev_ptr), A);

        if (debugz.dbg.at(.info)) debugz.dbg.printLevel(.info, "[gpu-vit] blk {d}: FFN\n", .{il});
        // ── 6. LN2 → FFN → residual 2 (out queda en d_x)
        if (self.use_rms_norm) {
            try self.lk.rmsNorm(@intFromPtr(self.d_inp_l.dev_ptr), @intFromPtr(b.ln2_w.dev_ptr), @intFromPtr(self.d_ln2.dev_ptr), n, ne, self.eps);
        } else {
            try self.lk.layerNormDev(@intFromPtr(self.d_inp_l.dev_ptr), @intFromPtr(b.ln2_w.dev_ptr), @intFromPtr(b.ln2_b.dev_ptr), @intFromPtr(self.d_ln2.dev_ptr), n, ne, self.eps);
        }
        try cublas.gemmF32DeviceResident(self.handle, self.d_ln2.dev_ptr, b.ff_up.dev_ptr, self.d_mid.dev_ptr, n, self.n_ff, ne, false, true);
        try self.lk.biasAddDev(@intFromPtr(self.d_mid.dev_ptr), @intFromPtr(b.ff_up_b.dev_ptr), n * self.n_ff, self.n_ff);
        try self.lk.geluDev(@intFromPtr(self.d_mid.dev_ptr), n * self.n_ff);
        try cublas.gemmF32DeviceResident(self.handle, self.d_mid.dev_ptr, b.ff_down.dev_ptr, self.d_ffn.dev_ptr, n, ne, self.n_ff, false, true);
        try self.lk.biasAddDev(@intFromPtr(self.d_ffn.dev_ptr), @intFromPtr(b.ff_down_b.dev_ptr), A, ne);

        // residual 2: out = inp_l + ffn (en d_x)
        try self.lk.add(@intFromPtr(self.d_inp_l.dev_ptr), @intFromPtr(self.d_ffn.dev_ptr), @intFromPtr(self.d_x.dev_ptr), A);

        // Deepstack (Qwen3-VL): si este bloque tiene deepstack, compute
        // d_ds_out = fc2(GELU(fc1(LayerNorm(shuffle(d_x))))) donde
        // shuffle: [n_pos, n_embd] -> [n_pos, n_embd*merge2].
        if (self.has_deepstack and il == self.ds_block_idx) {
            const dsw = &self.ds[0];
            const ne4 = self.n_embd * 4;
            try self.lk.deepstackShuffle(@intFromPtr(self.d_x.dev_ptr), @intFromPtr(self.d_ds_view.dev_ptr), self.n_pos, self.n_embd, self.n_pos, 4);
            if (self.use_rms_norm) {
                try self.lk.rmsNorm(@intFromPtr(self.d_ds_view.dev_ptr), @intFromPtr(dsw.norm_w.dev_ptr), @intFromPtr(dsw.norm_b.dev_ptr), self.n_pos, ne4, self.eps);
            } else {
                try self.lk.layerNormDev(@intFromPtr(self.d_ds_view.dev_ptr), @intFromPtr(dsw.norm_w.dev_ptr), @intFromPtr(dsw.norm_b.dev_ptr), @intFromPtr(self.d_ln2.dev_ptr), self.n_pos, ne4, self.eps);
                try self.lk.add(@intFromPtr(self.d_ds_view.dev_ptr), @intFromPtr(self.d_ln2.dev_ptr), @intFromPtr(self.d_ds_view.dev_ptr), self.n_pos * ne4);
            }
            try cublas.gemmF32DeviceResident(self.handle, self.d_ds_view.dev_ptr, dsw.fc1_w.dev_ptr, self.d_mid.dev_ptr, self.n_pos, dsw.n_ff, ne4, false, true);
            try self.lk.biasAddDev(@intFromPtr(self.d_mid.dev_ptr), @intFromPtr(dsw.fc1_b.dev_ptr), self.n_pos * dsw.n_ff, dsw.n_ff);
            try self.lk.geluDev(@intFromPtr(self.d_mid.dev_ptr), self.n_pos * dsw.n_ff);
            try cublas.gemmF32DeviceResident(self.handle, self.d_mid.dev_ptr, dsw.fc2_w.dev_ptr, self.d_ds_out.dev_ptr, self.n_pos, dsw.ds_out_dim, dsw.n_ff, false, true);
            try self.lk.biasAddDev(@intFromPtr(self.d_ds_out.dev_ptr), @intFromPtr(dsw.fc2_b.dev_ptr), self.n_pos * dsw.ds_out_dim, dsw.ds_out_dim);
        }
    }

    /// Ejecuta TODO el stack de bloques en GPU: d_x (input subido) →
    /// forwardBlock por cada capa → resultado en d_x.
    pub fn runBlocks(self: *Self) !void {
        for (0..self.blocks.len) |il| {
            try self.forwardBlock(il);
        }
    }

    /// Sube los pos-ids M-RoPE del ViT (visionPosIds del host).
    pub fn setPosIds(self: *Self, ids: []const [4]i32) !void {
        @memcpy(self.pos_ids_host, ids);
        if (self.d_pos_ids == 0) {
            self.d_pos_ids = try cudaz.cuMemAlloc(self.n_pos * 4 * @sizeOf(i32));
        }
        try cudaz.cuMemcpyHtoD(self.d_pos_ids, @intFromPtr(self.pos_ids_host.ptr), self.n_pos * 4 * @sizeOf(i32));
    }

    /// Sube la imagen preprocesada (input del bloque 0) a d_x.
    pub fn uploadInput(self: *Self, patches: []const f32) !void {
        try self.d_x.upload(patches);
    }

    /// Construye desde un ClipEncoder HOST (anytype: los ClipBlock ya
    /// tienen los pesos dequantizados). El GpuClipEncoder NO toma ownership
    /// de los pesos host (sólo los copia a device una vez).
    pub fn fromEncoder(
        allocator: std.mem.Allocator,
        lk: *layer_kernels.LayerKernels,
        enc: anytype, // *const ClipEncoder
        n_pos: usize,
    ) !Self {
        // pesos host por bloque desde los ClipBlock del encoder
        const n_blocks = enc.blocks.len;
        const bh = try allocator.alloc(BlockHostWeights, n_blocks);
        defer allocator.free(bh);
        for (enc.blocks, 0..) |*blk, i| {
            const empty_bias = [_]f32{};
            bh[i] = .{
                .qkv_w = blk.attn.qkv_w_t.data,
                .qkv_b = blk.attn.qkv_b orelse empty_bias[0..],
                .o_w = blk.attn.o_w_t.data,
                .o_b = blk.attn.o_b orelse empty_bias[0..],
                .ln1_w = blk.ln1_w,
                .ln1_b = blk.ln1_b orelse empty_bias[0..],
                .ln2_w = blk.ln2_w,
                .ln2_b = blk.ln2_b orelse empty_bias[0..],
                .ff_up = blk.ff_up_t.data,
                .ff_up_b = blk.ff_up_b orelse empty_bias[0..],
                .ff_down = blk.ff_down_t.data,
                .ff_down_b = blk.ff_down_b orelse empty_bias[0..],
            };
        }
        // Deepstack weights (si algún bloque los tiene)
        const ds_list = blk: {
            var count: usize = 0;
            for (enc.blocks) |*blk| {
                if (blk.ds) |_| count += 1;
            }
            if (count == 0) break :blk &[_]DeepstackHostWeights{};
            const arr = try allocator.alloc(DeepstackHostWeights, count);
            var idx: usize = 0;
            for (enc.blocks, 0..) |*blk, i| {
                if (blk.ds) |*ds| {
                    const ds_in = ds.fc1_t.shape[1]; // [n_ff, n_embd*4]
                    const ds_out = ds.fc2_t.shape[0]; // [ds_out_dim, n_ff]
                    arr[idx] = .{
                        .norm_w = ds.norm_w,
                        .norm_b = ds.norm_b orelse &.{},
                        .fc1_w = ds.fc1_t.data,
                        .fc1_b = ds.fc1_b orelse &.{},
                        .fc2_w = ds.fc2_t.data,
                        .fc2_b = ds.fc2_b orelse &.{},
                        .n_ff = ds.n_ff,
                        .ds_in = ds_in,
                        .ds_out_dim = ds_out,
                        .block_idx = i,
                    };
                    idx += 1;
                }
             }
             break :blk arr;
         };
         return Self.init(allocator, lk, bh, enc.n_embd, enc.blocks[0].n_ff, enc.n_head, enc.head_dim, n_pos, enc.use_rms_norm, enc.eps, ds_list);
     }

    /// Baja el resultado final (d_x tras el último bloque) a host.
    pub fn downloadOutput(self: *Self, out: []f32) !void {
        self.d_x.download(out) catch return ClipGpuError.CudaUnavailable;
    }

    /// Baja las deepstack features (d_ds_out) a host.
    pub fn downloadDsOutput(self: *Self, out: []f32) !void {
        if (!self.has_deepstack) return;
        self.d_ds_out.download(out) catch return ClipGpuError.CudaUnavailable;
    }
};
