//! KvarnGpuCache (lane-b1 Dev-A, M3/P4): ownership device-side de los
//! buffers KVarN por capa — records (C1), stage (C2v2), indices y los
//! descriptores K/V que consume el dispatcher (B7).
//!
//! Contrato con M3 (Dev-B M3_INTEGRATION §4.4): el token loop llama
//! `appendTokens` tras proyectar K/V del token (f32 device, dominio
//! ORIGINAL — el store rota), y luego `descs(layer)` alimenta
//! `dispatchKvarnAttentionSplit`. Los buffers viven por capa para
//! toda la vida de la secuencia; el alloc es lazy por capa (solo las
//! capas efectivamente usadas consumen VRAM).
//!
//! Layout device:
//!   records: [groups_per_stream × n_kv_heads × tile_bytes] por capa
//!   stage:   [stage_groups × 128 × (2·n_kv_heads) × head_dim/1? ] —
//!            C2v2 filas interleaved K/V (2·n_record_heads filas por
//!            pos), see kvarn_desc.cuh kvarn_stage_row.
//!   indices: [n_tokens] i64 codificados
//!   descs:   [2·n_kv_heads] KvarnDesc (par K/V por head físico)
//!
//! All byte sizes mirror kvarn.zig tileLayout + KvarnRecordLayout.

const std = @import("std");
const kvarn = @import("kv_cache").kvarn; // C1 de B2 (misma dep que kvarn_kernels)
pub const kvarn_types = @import("kv_cache").kvarn; // re-export M3
const cudaz = @import("cudaz");
const kvk = @import("kvarn_kernels");

/// Configuración por modelo (constante durante la vida del cache).
pub const KvarnGpuConfig = struct {
    smem_optin: ?u32 = null, // ver kvarnStoreAdaptive (A5)
    num_layers: u32,
    n_kv_heads: u32,
    head_dim: u32, // 64/128/256 (9.13; 512 pendiente store D-slice)
    k_bits: u8 = 5,
    v_bits: u8 = 4,
    /// Máximo contexto en tokens — determina groups_per_stream.
    max_ctx_tokens: u32,
    /// Groups residentes en stage (sink + tail window). Upstream 4.
    stage_groups: u32 = 4,
    tail_groups: u32 = 3,
    eager_records: u8 = 1,
    swa: bool = false,
};

/// Buffers device de UNA capa.
pub const LayerGpuState = struct {
    records: cudaz.CUdeviceptr,
    stage: cudaz.CUdeviceptr,
    indices: cudaz.CUdeviceptr,
    descs: cudaz.CUdeviceptr, // [2·n_kv_heads] KvarnDesc
    n_tokens: u32 = 0, // tokens ingested (drives indices/encoder)
    initialized: bool = false,
};

pub const KvarnGpuCache = struct {
    allocator: std.mem.Allocator,
    config: KvarnGpuConfig,
    layout: kvarn.KvarnRecordLayout,
    layers: []?LayerGpuState,
    module: ?cudaz.CUmodule = null, // kvarn cubin (store/init)
    smem_optin: ?u32 = null, // A5: policy hishmem/lowshmem
    stream: ?cudaz.CUstream = null,

    const Self = @This();

    /// 9.4 (lane-b) D2: cabezas FÍSICAS por slice — D=256 despliega
    /// 2 físicas de 128 por lógica (records/stage dimensionados con
    /// éstas; el layout del record es SIEMPRE 128). D=128: 1.
    /// 9.13 (rlt): D=64 = 1 slice físico (record rect 64, stage inner 128).
    pub fn headSlices(self: *const Self) u32 {
        return @max(1, self.config.head_dim / 128);
    }

    /// Heads FÍSICAS totales (n_kv_heads lógicas × slices).
    pub fn physicalHeads(self: *const Self) u32 {
        return self.config.n_kv_heads * self.headSlices();
    }

    /// Cubin paths vienen de build_options (main.zig los pasa):
    /// `kvarn_cubin` (store/initDescs). Sin cubin ⇒ error.NoCubin.
    pub fn init(
        allocator: std.mem.Allocator,
        config: KvarnGpuConfig,
        module: ?cudaz.CUmodule, // null ⇒ cache latente hasta attachCubin
        stream: cudaz.CUstream,
    ) !Self {
        // 9.4 (lane-b) D2: D=256 ratificado (store d256 + portable fixes
        // @63a4421); D=512 requiere store D-slice propio — gated.
        // 9.13 (rlt): D=64 ratificado (store d64 rect + fattn portable
        // D64, kernels 9.12 F2/F3 @31dbe1a + BUG A @15a4b5c).
        if (config.head_dim != 64 and config.head_dim != 128 and config.head_dim != 256) return error.UnsupportedHeadDim;
        if (config.n_kv_heads == 0 or config.num_layers == 0)
            return error.InvalidConfig;
        // Layout del record: 128 por cabeza física en D=128/256; D=64 usa
        // tile rect propio (layout 64) — 9.13 (rlt).
        const layout = try kvarn.KvarnRecordLayout.init(if (config.head_dim == 64) 64 else 128, config.k_bits, config.v_bits);
        const layers = try allocator.alloc(?LayerGpuState, config.num_layers);
        @memset(layers, null);
        return .{
            .allocator = allocator,
            .config = config,
            .layout = layout,
            .layers = layers,
            .module = module,
            .stream = stream,
            .smem_optin = config.smem_optin,
        };
    }

    /// M3 slice 2 (Dev-B): fija el cubin tras init-latente (module=null
    /// en init). Llamado tras cuModuleLoad(build_options.kvarn_cubin).
    pub fn attachCubin(self: *Self, module: cudaz.CUmodule) void {
        self.module = module;
    }

    pub fn deinit(self: *Self) void {
        for (self.layers) |maybe| {
            if (maybe) |st| {
                cudaz.cuMemFree(st.records);
                cudaz.cuMemFree(st.stage);
                cudaz.cuMemFree(st.indices);
                cudaz.cuMemFree(st.descs);
            }
        }
        self.allocator.free(self.layers);
    }

    pub fn groupsPerStream(self: *const Self) u32 {
        // tokens = groups·128; alloc para max_ctx redondeado a grupo.
        const gps = (self.config.max_ctx_tokens + kvarn.KVAR_N_GROUP - 1) / kvarn.KVAR_N_GROUP;
        return @max(gps, 1);
    }

    pub fn stageLen(self: *const Self) usize {
        // C2v2: [stage_groups][128 pos][2·physical_heads filas][128 dims] f16.
        // 9.4 (lane-b) D2: D=256 ⇒ physical_heads = 2·n_kv_heads.
        return @as(usize, self.config.stage_groups) * kvarn.KVAR_N_GROUP *
            (2 * @as(usize, self.physicalHeads())) * 128;
    }

    pub fn indicesLen(self: *const Self) usize {
        return @as(usize, self.groupsPerStream()) * kvarn.KVAR_N_GROUP;
    }

    pub fn recordsLen(self: *const Self) usize {
        // Records por cabeza FÍSICA (layout 128), D=256 ⇒ 2·n_kv_heads.
        return @as(usize, self.groupsPerStream()) *
            @as(usize, self.physicalHeads()) * self.layout.tile_bytes;
    }

    pub fn descsLen(self: *const Self) usize {
        // Un par (K,V) por kv_head LÓGICA — head_base del desc apunta a
        // la base de sus slices físicas.
        return 2 * @as(usize, self.config.n_kv_heads) * @sizeOf(kvk.KvarnDesc);
    }

    /// Lazy device alloc de la capa (idempotente).
    fn ensureLayer(self: *Self, layer: u32) !*LayerGpuState {
        if (layer >= self.config.num_layers) return error.LayerOutOfRange;
        if (self.layers[layer] == null) {
            // Zero-init para que records/stage/descs estén limpios.
            const records = try cudaz.cuMemAlloc(self.recordsLen());
            errdefer cudaz.cuMemFree(records);
            const stage = try cudaz.cuMemAlloc(@sizeOf(f16) * self.stageLen());
            errdefer cudaz.cuMemFree(stage);
            const indices = try cudaz.cuMemAlloc(@sizeOf(i64) * self.indicesLen());
            errdefer cudaz.cuMemFree(indices);
            const descs = try cudaz.cuMemAlloc(self.descsLen());
            errdefer cudaz.cuMemFree(descs);
            try cudaz.cuMemsetD8(records, 0, self.recordsLen());
            try cudaz.cuMemsetD8(stage, 0, @sizeOf(f16) * self.stageLen());
            try cudaz.cuMemsetD8(indices, 0xFF, @sizeOf(i64) * self.indicesLen());
            try cudaz.cuMemsetD8(descs, 0, self.descsLen());
            self.layers[layer] = .{
                .records = records,
                .stage = stage,
                .indices = indices,
                .descs = descs,
            };
        }
        return &self.layers[layer].?;
    }

    /// Ingresa n tokens de UNA capa: current_k/current_v son f32
    /// device [n × n_kv_heads × 128] dominio ORIGINAL (el store rota).
    /// indices_host: celdas codificadas (o null ⇒ secuencia directa
    /// [base..base+n)). `base` = posición absoluta de esta tanda.
    pub fn appendTokens(
        self: *Self,
        layer: u32,
        current_k: cudaz.CUdeviceptr,
        current_v: ?cudaz.CUdeviceptr,
        n_tokens: u32,
        base_token: u32,
        indices_host: ?[]const i64,
        stream: cudaz.CUstream,
    ) !void {
        const st = try self.ensureLayer(layer);
        const module = self.module orelse return error.NoCubin;
        const new_base = base_token + n_tokens;
        if (new_base > self.indicesLen()) return error.ContextOverflow;

        // Indices: si el caller provee celdas codificadas, subirlas;
        // si no, la porción [base..base+n) se llena directa.
        if (indices_host) |cells| {
            if (cells.len != n_tokens) return error.IndicesLenMismatch;
            const staging = try self.allocator.alignedAlloc(u8, .@"8", @sizeOf(i64) * n_tokens);
            defer self.allocator.free(staging);
            const dst: [*]i64 = @ptrCast(@alignCast(staging.ptr));
            @memcpy(dst[0..n_tokens], cells);
            try cudaz.cuMemcpyHtoD(st.indices + @as(usize, base_token) * @sizeOf(i64), @intFromPtr(&dst[0]), @sizeOf(i64) * n_tokens);
        } else {
            const staging = try self.allocator.alignedAlloc(u8, .@"8", @sizeOf(i64) * n_tokens);
            defer self.allocator.free(staging);
            const dst: [*]i64 = @ptrCast(@alignCast(staging.ptr));
            for (0..n_tokens) |i| dst[i] = @intCast(base_token + i);
            try cudaz.cuMemcpyHtoD(st.indices + @as(usize, base_token) * @sizeOf(i64), @intFromPtr(&dst[0]), @sizeOf(i64) * n_tokens);
        }

        // 9.4 (lane-b) D2: D=256 usa el store D-slice (cross-slice WHT +
        // cabezas físicas); D=128 el store clásico (invariante M3).
        // 9.13 (rlt): D=64 usa el store rect (kvarn_store_d64_kernel,
        // tiles K 64×128 / V 128×64, stage inner 128 con first64).
        // 9.4 FIX incremental: el kernel itera indices[0..n_tokens) — con
        // base>0 hay que apuntar al SLICE base..base+n (si no, el decode
        // n=1/base=5 lee indices[0]=0 y SOBRESCRIBE el token 0 del stage).
        if (self.headSlices() == 2) {
            var dargs: kvk.KvarnStoreD256Args = .{
                .current = @ptrFromInt(current_k),
                .current_v = if (current_v) |v| @ptrFromInt(v) else null,
                .indices = @ptrFromInt(st.indices + @as(usize, base_token) * @sizeOf(i64)),
                .stage = @ptrFromInt(st.stage),
                .records = @ptrFromInt(st.records),
                .n_tokens = @intCast(n_tokens),
                .n_logical_heads = @intCast(self.config.n_kv_heads),
                .n_record_heads = @intCast(self.physicalHeads()),
                .stream = 0,
                .groups_per_stream = @intCast(self.groupsPerStream()),
                .record_bytes = @intCast(self.layout.tile_bytes),
                .k_payload_off = @intCast(self.layout.k_payload_off),
                .k_s_col_off = @intCast(self.layout.k_s_col_off),
                .k_zp_off = @intCast(self.layout.k_zp_off),
                .k_s_row_off = @intCast(self.layout.k_s_row_off),
                .v_payload_off = @intCast(self.layout.v_payload_off),
                .v_s_col_off = @intCast(self.layout.v_s_col_off),
                .v_s_row_off = @intCast(self.layout.v_s_row_off),
                .v_zp_off = @intCast(self.layout.v_zp_off),
                .k_bits = self.config.k_bits,
                .v_bits = self.config.v_bits,
                .sinkhorn_iters = 16,
                .stage_groups = @intCast(self.config.stage_groups),
                .tail_groups = @intCast(self.config.tail_groups),
                .swa = if (self.config.swa) 1 else 0,
                .eager_records = self.config.eager_records,
            };
            try kvk.kvarnStoreD256Device(module, &dargs, stream);
        } else if (self.config.head_dim == 64) {
            var d64args: kvk.KvarnStoreD64Args = .{
                .current = @ptrFromInt(current_k),
                .current_v = if (current_v) |v| @ptrFromInt(v) else null,
                .indices = @ptrFromInt(st.indices + @as(usize, base_token) * @sizeOf(i64)),
                .stage = @ptrFromInt(st.stage),
                .records = @ptrFromInt(st.records),
                .n_tokens = @intCast(n_tokens),
                .n_record_heads = @intCast(self.physicalHeads()),
                .stream = 0,
                .groups_per_stream = @intCast(self.groupsPerStream()),
                .record_bytes = @intCast(self.layout.tile_bytes),
                .k_payload_off = @intCast(self.layout.k_payload_off),
                .k_s_col_off = @intCast(self.layout.k_s_col_off),
                .k_zp_off = @intCast(self.layout.k_zp_off),
                .k_s_row_off = @intCast(self.layout.k_s_row_off),
                .v_payload_off = @intCast(self.layout.v_payload_off),
                .v_s_col_off = @intCast(self.layout.v_s_col_off),
                .v_s_row_off = @intCast(self.layout.v_s_row_off),
                .v_zp_off = @intCast(self.layout.v_zp_off),
                .k_bits = self.config.k_bits,
                .v_bits = self.config.v_bits,
                .sinkhorn_iters = 16,
                .stage_groups = @intCast(self.config.stage_groups),
                .tail_groups = @intCast(self.config.tail_groups),
                .swa = if (self.config.swa) 1 else 0,
                .eager_records = self.config.eager_records,
            };
            try kvk.kvarnStoreD64Device(module, &d64args, stream);
        } else {
            var sargs: kvk.KvarnStoreArgs = .{
                .current = @ptrFromInt(current_k),
                .current_v = if (current_v) |v| @ptrFromInt(v) else null,
                .indices = @ptrFromInt(st.indices),
                .stage = @ptrFromInt(st.stage),
                .records = @ptrFromInt(st.records),
                .n_tokens = @intCast(n_tokens),
                .n_record_heads = @intCast(self.config.n_kv_heads),
                .stream = 0,
                .groups_per_stream = @intCast(self.groupsPerStream()),
                .record_bytes = @intCast(self.layout.tile_bytes),
                .k_payload_off = @intCast(self.layout.k_payload_off),
                .k_s_col_off = @intCast(self.layout.k_s_col_off),
                .k_zp_off = @intCast(self.layout.k_zp_off),
                .k_s_row_off = @intCast(self.layout.k_s_row_off),
                .v_payload_off = @intCast(self.layout.v_payload_off),
                .v_s_col_off = @intCast(self.layout.v_s_col_off),
                .v_s_row_off = @intCast(self.layout.v_s_row_off),
                .v_zp_off = @intCast(self.layout.v_zp_off),
                .k_bits = self.config.k_bits,
                .v_bits = self.config.v_bits,
                .sinkhorn_iters = 16,
                .stage_groups = @intCast(self.config.stage_groups),
                .tail_groups = @intCast(self.config.tail_groups),
                .swa = if (self.config.swa) 1 else 0,
                .eager_records = self.config.eager_records,
            };
            try kvk.kvarnStoreAdaptive(module, &sargs, stream, self.smem_optin);
        }
        st.n_tokens = new_base;

        // Refresh descs (live tracking tras el store).
        try self.refreshDescs(layer, stream);
    }

    /// Recalcula los descriptores de la capa (tras append o restore).
    pub fn refreshDescs(self: *Self, layer: u32, stream: cudaz.CUstream) !void {
        const st = try self.ensureLayer(layer);
        const module = self.module orelse return error.NoCubin;
        var ia: kvk.KvarnInitDescsArgs = .{
            .n_stream = 1,
            .n_indices = @intCast(st.n_tokens),
            .d_indices = @ptrFromInt(st.indices),
            .d_descs = @ptrFromInt(st.descs),
            // 9.4 (lane-b): layout por-lado [K(n), V(n)] por stream —
            // stride = 2·n_kv_heads lógicas (bloque K + bloque V).
            .desc_stride = @intCast(2 * self.config.n_kv_heads),
            .d_records = @ptrFromInt(st.records),
            .d_stage = @ptrFromInt(st.stage),
            // 9.4 (lane-b) D2: heads FÍSICAS (2·n_kv_heads con D=256) —
            // el loop del init itera n_kv_heads LÓGICAS (= físicas/slices).
            .n_record_heads = @intCast(self.physicalHeads()),
            .head_dim = @intCast(self.config.head_dim),
            .groups_per_stream = @intCast(self.groupsPerStream()),
            .record_bytes = @intCast(self.layout.tile_bytes),
            .stage_groups = @intCast(self.config.stage_groups),
            .tail_groups = @intCast(self.config.tail_groups),
            .k_bits = self.config.k_bits,
            .v_bits = self.config.v_bits,
            .head_slices = @intCast(self.headSlices()),
            .eager_records = self.config.eager_records,
            .read_indirect = 0,
            .original_domain = 0,
            .swa = if (self.config.swa) 1 else 0,
        };
        try kvk.kvarnInitDescsDevice(module, &ia, stream);
        st.initialized = true;
    }

    /// Descriptors para el dispatcher: base K y base V de la capa.
    /// (k = descs; v = descs + sizeOf(KvarnDesc) — layout par K/V por
    /// head físico.)
    pub fn kDescs(self: *const Self, layer: u32) !cudaz.CUdeviceptr {
        const st = self.layers[layer] orelse return error.LayerNotAllocated;
        return st.descs;
    }

    pub fn vDescs(self: *const Self, layer: u32) !cudaz.CUdeviceptr {
        const st = self.layers[layer] orelse return error.LayerNotAllocated;
        return st.descs + @sizeOf(kvk.KvarnDesc);
    }

    pub fn tokenCount(self: *const Self, layer: u32) u32 {
        if (self.layers[layer]) |st| return st.n_tokens;
        return 0;
    }
};

test "KvarnGpuCache: config errors pre-GPU" {
    // Sin GPU necesaria: validación de config pura.
    const allocator = std.testing.allocator;
    const c1 = KvarnGpuConfig{
        .num_layers = 4,
        .n_kv_heads = 2,
        .head_dim = 512, // 9.13: 64/128/256 soportados; 512 gated (store D-slice)
        .k_bits = 5,
        .v_bits = 4,
        .max_ctx_tokens = 2048,
    };
    // head_dim no soportado ⇒ UnsupportedHeadDim ANTES de tocar CUDA:
    // init requiere module/stream — usamos undefined: la validación
    // corre antes de cualquier llamada device.
    _ = try std.testing.expectError(error.UnsupportedHeadDim, KvarnGpuCache.init(allocator, c1, undefined, undefined));
    const c2 = KvarnGpuConfig{ .num_layers = 0, .n_kv_heads = 2, .head_dim = 128, .k_bits = 5, .v_bits = 4, .max_ctx_tokens = 2048 };
    _ = try std.testing.expectError(error.InvalidConfig, KvarnGpuCache.init(allocator, c2, undefined, undefined));
}
