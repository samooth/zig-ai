//! Motor GPU de PagedAttention (decode / prefill / reshape / copy).
//! Carga el cubin `paged_attention.cubin` compilado por el build (para la
//! arquitectura GPU detectada) y lanza los kernels equivalentes a la referencia
//! CPU (`attention.zig`).
//! Layout de bloques: memory-pool de `BlockAllocator` (K region + V region por
//! bloque físico), el mismo que lee `PagedAttention.decode`.
const std = @import("std");
const cudaz = @import("cudaz");
const debug = @import("debug");
const build_options = @import("build_options");
const BlockTable = @import("block_table.zig").BlockTable;
const BlockAllocator = @import("allocator.zig").BlockAllocator;
const PagedGpuBlockPool = @import("paged_gpu_pool.zig").PagedGpuBlockPool;
const PagedConfig = @import("root.zig").PagedConfig;
const QuantFormat = @import("root.zig").QuantFormat;

/// Dispatch table entry for a quantized kernel
const KernelDispatch = struct {
    kernel_name: []const u8,
    /// Number of scale arrays per block (K scales + V scales)
    scale_arrays_per_block: usize,
    /// Whether this uses 256-element super-blocks (K-quants, I-quants, T-quants)
    super_block: bool,
    /// Bytes per super-block (for pointer arithmetic in kernel)
    bytes_per_super_block: usize,
};

/// Initialize the kernel dispatch table with all supported formats
fn initKernelDispatch(_: std.mem.Allocator) !std.AutoHashMap(QuantFormat, KernelDispatch) {
    // Sin defer: el mapa se devuelve por valor y su copia vive como caché
    // global (getKernelDispatch). Un deinit aquí colgaría los punteros
    // internos compartidos por la copia (use-after-free en el primer .get).
    // page_allocator: caché global que vive todo el proceso; el testing
    // allocator detectaría el "leak" intencional al fin del test.
    var map = std.AutoHashMap(QuantFormat, KernelDispatch).init(std.heap.page_allocator);

    // 32-element block formats (legacy)
    // q8_0: 34 bytes/block, 1 scale array for K, 1 for V
    try map.put(.q8_0, .{ .kernel_name = "paged_attention_decode_q8_0_kernel", .scale_arrays_per_block = 2, .super_block = false, .bytes_per_super_block = 0 });
    // q4_0: 18 bytes/block, 1 scale array for K, 1 for V
    try map.put(.q4_0, .{ .kernel_name = "paged_attention_decode_q4_0_kernel", .scale_arrays_per_block = 2, .super_block = false, .bytes_per_super_block = 0 });
    // q4_1: 20 bytes/block, 1 scale array for K, 1 for V
    try map.put(.q4_1, .{ .kernel_name = "paged_attention_decode_q4_1_kernel", .scale_arrays_per_block = 2, .super_block = false, .bytes_per_super_block = 0 });
    // q5_0: 22 bytes/block
    try map.put(.q5_0, .{ .kernel_name = "paged_attention_decode_q5_0_kernel", .scale_arrays_per_block = 2, .super_block = false, .bytes_per_super_block = 0 });
    // q5_1: 24 bytes/block
    try map.put(.q5_1, .{ .kernel_name = "paged_attention_decode_q5_1_kernel", .scale_arrays_per_block = 2, .super_block = false, .bytes_per_super_block = 0 });
    // q8_1: 36 bytes/block
    try map.put(.q8_1, .{ .kernel_name = "paged_attention_decode_q8_1_kernel", .scale_arrays_per_block = 2, .super_block = false, .bytes_per_super_block = 0 });
    // int4: 32+8 bytes/block
    try map.put(.int4, .{ .kernel_name = "paged_attention_decode_int4_kernel", .scale_arrays_per_block = 2, .super_block = false, .bytes_per_super_block = 0 });
    // int8_symmetric: 64+4 bytes/block
    try map.put(.int8_symmetric, .{ .kernel_name = "paged_attention_decode_int8_symmetric_kernel", .scale_arrays_per_block = 2, .super_block = false, .bytes_per_super_block = 0 });
    // int8_asymmetric: 64+8 bytes/block
    try map.put(.int8_asymmetric, .{ .kernel_name = "paged_attention_decode_int8_asymmetric_kernel", .scale_arrays_per_block = 2, .super_block = false, .bytes_per_super_block = 0 });

    // 256-element super-block formats (K-quants)
    // q2_k: 84 bytes/super-block
    try map.put(.q2_k, .{ .kernel_name = "paged_attention_decode_q2_k_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 84 });
    // q3_k: 110 bytes/super-block
    try map.put(.q3_k, .{ .kernel_name = "paged_attention_decode_q3_k_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 110 });
    // q4_k: 144 bytes/super-block (already has kernel)
    try map.put(.q4_k, .{ .kernel_name = "paged_attention_decode_q4_k_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 144 });
    // q5_k: 176 bytes/super-block
    try map.put(.q5_k, .{ .kernel_name = "paged_attention_decode_q5_k_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 176 });
    // q6_k: 210 bytes/super-block
    try map.put(.q6_k, .{ .kernel_name = "paged_attention_decode_q6_k_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 210 });
    // q8_k: 292 bytes/super-block
    try map.put(.q8_k, .{ .kernel_name = "paged_attention_decode_q8_k_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 292 });

    // I-quants (256-element super-blocks, ternary codebook)
    // iq1_s: 50 bytes/super-block
    try map.put(.iq1_s, .{ .kernel_name = "paged_attention_decode_iq1_s_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 50 });
    // iq1_m: 56 bytes/super-block
    try map.put(.iq1_m, .{ .kernel_name = "paged_attention_decode_iq1_m_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 56 });
    // iq2_xxs: 66 bytes/super-block
    try map.put(.iq2_xxs, .{ .kernel_name = "paged_attention_decode_iq2_xxs_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 66 });
    // iq2_xs: 74 bytes/super-block
    try map.put(.iq2_xs, .{ .kernel_name = "paged_attention_decode_iq2_xs_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 74 });
    // iq2_s: 82 bytes/super-block
    try map.put(.iq2_s, .{ .kernel_name = "paged_attention_decode_iq2_s_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 82 });
    // iq3_xxs: 98 bytes/super-block
    try map.put(.iq3_xxs, .{ .kernel_name = "paged_attention_decode_iq3_xxs_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 98 });
    // iq3_s: 110 bytes/super-block
    try map.put(.iq3_s, .{ .kernel_name = "paged_attention_decode_iq3_s_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 110 });
    // iq4_xs: 136 bytes/super-block
    try map.put(.iq4_xs, .{ .kernel_name = "paged_attention_decode_iq4_xs_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 136 });
    // iq4_nl: 18 bytes/32 elems (different - uses 32-element blocks!)
    try map.put(.iq4_nl, .{ .kernel_name = "paged_attention_decode_iq4_nl_kernel", .scale_arrays_per_block = 2, .super_block = false, .bytes_per_super_block = 0 });

    // T-quants (256-element super-blocks)
    // tq1_0: 54 bytes/super-block
    try map.put(.tq1_0, .{ .kernel_name = "paged_attention_decode_tq1_0_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 54 });
    // tq2_0: 66 bytes/super-block
    try map.put(.tq2_0, .{ .kernel_name = "paged_attention_decode_tq2_0_kernel", .scale_arrays_per_block = 2, .super_block = true, .bytes_per_super_block = 66 });

    // MXFP4 (32-element blocks)
    try map.put(.mxfp4, .{ .kernel_name = "paged_attention_decode_mxfp4_kernel", .scale_arrays_per_block = 2, .super_block = false, .bytes_per_super_block = 0 });

    return map;
}

/// Global dispatch table (initialized on first use)
var kernel_dispatch: ?std.AutoHashMap(QuantFormat, KernelDispatch) = null;

fn getKernelDispatch(allocator: std.mem.Allocator) !std.AutoHashMap(QuantFormat, KernelDispatch) {
    if (kernel_dispatch) |d| return d;
    const d = try initKernelDispatch(allocator);
    kernel_dispatch = d;
    return d;
}

pub const PagedAttentionGpuError = error{
    CudaUnavailable,
    KernelNotFound,
    KvQuantUnsupported,
};

/// Pool de bloques persistente en el dispositivo. Aloca el buffer `d_cache`
/// una sola vez (`num_blocks * block_bytes`) y mantiene un bitmap de qué
/// bloques físicos están residentes en GPU. `stageBlock` sube bloques
/// individuales H2D; `evictBlock` los baja D2H de vuelta al host. Sustituye
/// las copias completas del memory-pool en cada llamada a `decode`.
pub const GpuBlockPool = struct {
    allocator: std.mem.Allocator,
    num_blocks: usize,
    block_bytes: usize,
    d_cache: cudaz.CUdeviceptr,
    resident: []bool,
    dirty: []bool,

    const Self = @This();

    pub fn init(gpa: std.mem.Allocator, num_blocks: usize, block_bytes: usize) !Self {
        const resident = try gpa.alloc(bool, num_blocks);
        errdefer gpa.free(resident);
        @memset(resident, false);
        const dirty = try gpa.alloc(bool, num_blocks);
        errdefer gpa.free(dirty);
        @memset(dirty, false);
        const d_cache = try cudaz.cuMemAlloc(num_blocks * block_bytes);
        return .{
            .allocator = gpa,
            .num_blocks = num_blocks,
            .block_bytes = block_bytes,
            .d_cache = d_cache,
            .resident = resident,
            .dirty = dirty,
        };
    }

    pub fn deinit(self: *Self) void {
        cudaz.cuMemFree(self.d_cache);
        self.allocator.free(self.resident);
        self.allocator.free(self.dirty);
    }

    pub fn markDirty(self: *Self, phys_id: usize) void {
        if (phys_id >= self.num_blocks) return;
        self.dirty[phys_id] = true;
    }

    /// Sube el bloque a GPU solo si no está residente o fue modificado en host
    /// (dirty). La copia del KV del token actual la marca dirty el forward de
    /// atención; los bloques pasados ya residentes se saltan (evita el H2D
    /// creciente con el contexto).
    pub fn stageBlock(self: *Self, block_alloc: *BlockAllocator, phys_id: usize) !void {
        if (phys_id >= self.num_blocks) return;
        if (self.resident[phys_id] and !self.dirty[phys_id]) return;
        const src = @intFromPtr(block_alloc.memory_pool.ptr) + phys_id * self.block_bytes;
        try cudaz.cuMemcpyHtoD(self.d_cache + phys_id * self.block_bytes, src, self.block_bytes);
        self.resident[phys_id] = true;
        self.dirty[phys_id] = false;
    }

    pub fn evictBlock(self: *Self, block_alloc: *BlockAllocator, phys_id: usize) !void {
        if (phys_id >= self.num_blocks or !self.resident[phys_id]) return;
        const dst = @intFromPtr(block_alloc.memory_pool.ptr) + phys_id * self.block_bytes;
        try cudaz.cuMemcpyDtoH(dst, self.d_cache + phys_id * self.block_bytes, self.block_bytes);
        self.resident[phys_id] = false;
    }

    /// Marca el bloque residente en device sin copiar H2D: el KV se escribe por
    /// GPU (kvAppendF16) directamente sobre `d_cache`. Para el pool contiguo el
    /// espacio ya está alocado, solo se actualiza el estado.
    pub fn ensureCommitted(self: *Self, phys_id: usize) void {
        if (phys_id >= self.num_blocks) return;
        self.resident[phys_id] = true;
        self.dirty[phys_id] = false;
    }

    /// Baja (async, stream-ordered) el bloque escrito por GPU al host pool.
    pub fn syncBlockToHost(self: *Self, block_alloc: *BlockAllocator, phys_id: usize, stream: cudaz.CUstream) !void {
        if (phys_id >= self.num_blocks) return;
        const dst = @intFromPtr(block_alloc.memory_pool.ptr) + phys_id * self.block_bytes;
        try cudaz.cuMemcpyDtoHAsync(dst, self.d_cache + phys_id * self.block_bytes, self.block_bytes, stream);
    }

    pub fn stageTable(self: *Self, block_alloc: *BlockAllocator, block_table: *const BlockTable) !void {
        debug.dbg.printLevel(.detail, "[gpu_block_pool] stageTable numBlocks={}\n", .{block_table.numBlocks()});
        for (0..block_table.numBlocks()) |i| {
            if (block_table.getPhysical(i)) |phys| {
                debug.dbg.printLevel(.detail, "[gpu_block_pool] staging block {} -> phys {}\n", .{ i, phys });
                try self.stageBlock(block_alloc, phys);
                debug.dbg.printLevel(.detail, "[gpu_block_pool] staged block {} -> phys {}\n", .{ i, phys });
            }
        }
    }

    pub fn evictAll(self: *Self, block_alloc: *BlockAllocator) !void {
        for (0..self.num_blocks) |phys| {
            try self.evictBlock(block_alloc, phys);
        }
    }

    /// Baja del dispositivo una lista de bloques fríos (D2H) y los marca no
    /// residentes. Se usa cuando la prefix cache desaloja entradas frías.
    pub fn evictBlocks(self: *Self, block_alloc: *BlockAllocator, phys_ids: []const usize) !void {
        for (phys_ids) |phys| {
            try self.evictBlock(block_alloc, phys);
        }
    }

    pub fn numResident(self: *const Self) usize {
        var n: usize = 0;
        for (self.resident) |r| {
            if (r) n += 1;
        }
        return n;
    }
};

const DecodePersistentParams = struct {
    num_seqs_c: c_int = 0,
    max_blocks_c: c_int = 0,
    num_q_c: c_int = 0,
    num_kv_c: c_int = 0,
    head_dim_c: c_int = 0,
    block_size_c: c_int = 0,
    d_cache_v: cudaz.CUdeviceptr = 0,
    q16v: cudaz.CUdeviceptr = 0,
    out16v: cudaz.CUdeviceptr = 0,
    // Escalas por super-bloque (kernels cuantizados; sin uso en f16).
    d_k_scales_v: cudaz.CUdeviceptr = 0,
    d_v_scales_v: cudaz.CUdeviceptr = 0,
    // G1 split-K: storage ESTABLE para kernelParams del launch split
    // (graph-safe: el grafo congela las DIRECCIONES; los valores se leen
    // de aquí en cada replay — nunca punteros al stack del caller).
    split_partials_v: cudaz.CUdeviceptr = 0,
    kp: [13]?*anyopaque = undefined,
    kp_split: [13]?*anyopaque = undefined,
    kp_split_combine: [5]?*anyopaque = undefined,
};
var g_decode_persistent: DecodePersistentParams = .{};

/// Kernel device→device disponible por formato + floats extra de shared memory
/// que su implementación requiere (escalas precargadas en smem). `null` → el
/// formato aún no tiene kernel fusionado en paged_attention.cu.
const DeviceDecodeKernel = struct { name: []const u8, extra_smem_floats: usize };

/// 3.3 (lane-f): variante dp4a del decode q8_0. A/B opt-in por env
/// ZIG_AI_PADP4A=1 (default OFF = kernel base byte-a-byte). El dot Q·K
/// corre por __dp4a con Q cuantizada a q8_0 en smem (4 int8-mults/instr).
/// smem extra: qb escalas + qb*8 u32 quanta + pad 16B (en f32 units).
/// 3.3-b NEGATIVO (datos): la variante q4_0 (extracción lo/hi de nibbles +
/// doble dp4a) es +16% MÁS LENTA en attn que el base (1914→2227 µs, 0.8B
/// q4_0 KV) — el K 4-bit ya es la mitad de bytes y el byte-dequant FMA no
/// era el cuello. Queda accesible SOLO para repro con ZIG_AI_PADP4A_Q4=1.
var padp4a_enabled: ?bool = null;
fn padp4aEnabled() bool {
    if (padp4a_enabled) |v| return v;
    padp4a_enabled = std.c.getenv("ZIG_AI_PADP4A") != null;
    return padp4a_enabled.?;
}

var padp4a_q4_enabled: ?bool = null;
fn padp4aQ4Enabled() bool {
    if (padp4a_q4_enabled) |v| return v;
    padp4a_q4_enabled = std.c.getenv("ZIG_AI_PADP4A_Q4") != null;
    return padp4a_q4_enabled.?;
}

/// G1 (lane-b1 2026-09-08): flash-decoding split-K del decode f16. A/B
/// opt-in por env ZIG_AI_FASPLIT=1 (default OFF = kernel base grid (1,H,1)
/// — con GQA el 0.8B son 8 warps totales: latency-bound). La variante
/// split lanza grid (1,H,n_splits)×128 con online-softmax parcial por
/// chunk + combine. Flip a default SOLO tras paridad greedy + bench ≥1.5×
/// (gate TODO 1.6).
var fasplit_enabled: ?bool = null;
fn fasplitEnabled() bool {
    if (fasplit_enabled) |v| return v;
    fasplit_enabled = std.c.getenv("ZIG_AI_FASPLIT") != null;
    return fasplit_enabled.?;
}

/// Tests (test_paged_attention_gpu): fuerza el estado del opt-in sin
/// depender del orden de tests en el binario (el cache ?bool se llena
/// una sola vez por proceso).
pub fn fasplitForceForTest(on: bool) void {
    fasplit_enabled = on;
}

/// G1 split-K: política de splits. n_splits FIJO por forma (num_q_heads×
/// head_dim del modelo) para que el buffer de partials y los escalares
/// congelados por el grafo nunca cambien de tamaño mid-session; el kernel
/// reparte tokens_per_split = ceil(seq_budget) y los splits sobrantes se
/// marcan vacíos (m=-inf) que el combine ignora. seq_budget parametriza
/// el sizing (presizeDecodeScratch pasa su budget; 0 = default).
fn splitPolicyForSeqLen(seq_budget: usize) usize {
    _ = seq_budget; // política por-forma por ahora; reservado para tuning por ctx
    return 8;
}

fn splitTokensPerSplit(seq_budget: usize) usize {
    _ = seq_budget;
    return 128; // 8 splits × 128 tokens = 1024 de cobertura nominal
}

fn deviceDecodeKernel(q: QuantFormat, elems_per_block: usize, head_dim: usize) ?DeviceDecodeKernel {
    const sb32 = 2 * ((elems_per_block + 31) / 32);
    const sb256 = 2 * ((elems_per_block + 255) / 256);
    return switch (q) {
        .fp16 => .{ .name = "paged_attention_decode_f16_kernel", .extra_smem_floats = 0 },
        // q8_0 reescrito: lee la escala embebida en el bloque de 34 bytes (sin smem extra).
        // 3.3: ZIG_AI_PADP4A=1 → variante dp4a; smem extra por head:
        // qb escalas d_q + qb*8 u32 quanta + pad de alineación 16B.
        .q8_0 => if (padp4aEnabled()) blk: {
            const qb = (head_dim + 31) / 32;
            break :blk .{ .name = "paged_attention_decode_q8_0_dp4a_kernel", .extra_smem_floats = qb * 9 + 4 };
        } else .{ .name = "paged_attention_decode_q8_0_kernel", .extra_smem_floats = 0 },
        .q4_0 => if (padp4aQ4Enabled()) blk: {
            // 3.3-b NEGATIVO (−16%): sólo para repro con ZIG_AI_PADP4A_Q4=1.
            const qb = (head_dim + 31) / 32;
            break :blk .{ .name = "paged_attention_decode_q4_0_dp4a_kernel", .extra_smem_floats = qb * 10 + 4 };
        } else .{ .name = "paged_attention_decode_q4_0_kernel", .extra_smem_floats = sb32 },
        // K-quants leen escalas inline desde los bytes del bloque (sin smem extra).
        .q4_k => .{ .name = "paged_attention_decode_q4_k_kernel", .extra_smem_floats = 0 },
        .q2_k => .{ .name = "paged_attention_decode_q2_k_kernel", .extra_smem_floats = 0 },
        .q3_k => .{ .name = "paged_attention_decode_q3_k_kernel", .extra_smem_floats = 0 },
        .q5_k => .{ .name = "paged_attention_decode_q5_k_kernel", .extra_smem_floats = 0 },
        .q6_k => .{ .name = "paged_attention_decode_q6_k_kernel", .extra_smem_floats = 0 },
        .q8_k => .{ .name = "paged_attention_decode_q8_k_kernel", .extra_smem_floats = sb256 },
        else => null,
    };
}

pub const PagedAttentionGpu = struct {
    allocator: std.mem.Allocator,
    config: PagedConfig,
    module: cudaz.CUmodule,
    /// Lane A: cubin secundario con formatos extra (iq4_xs, q4_1, q5_0/51, q8_1).
    /// null si build_options.fused_extra_cubin está vacío o el load falla.
    module_extra: ?cudaz.CUmodule = null,
    // Stream CUDA compartido con el resto de la capa híbrida (un solo stream,
    // una sola sincronización por token). NO se destruye en deinit.
    stream: cudaz.CUstream,
    pool: ?GpuBlockPool = null,
    paged_pool: ?PagedGpuBlockPool = null,
    // Buffers de decode persistentes (evitan cuMemAlloc/free por token).
    d_q16: cudaz.CUdeviceptr = 0,
    d_out16: cudaz.CUdeviceptr = 0,
    d_seq_lens: cudaz.CUdeviceptr = 0,
    d_start_pos: cudaz.CUdeviceptr = 0,

    // Staging host persistentes (fuentes de los HtoDAsync capturados por el
    // grafo de decode): se rellenan por token; el grafo los copia a device.
    // PINNED (cuMemAllocHost): los HtoDAsync con fuente pageable no son
    // capturables por CUDA graphs.
    // Cada capa de atención tiene SU PROPIA block table física (bloques
    // distintos del pool compartido), así que staging y d_bt son por capa:
    // indexados por layer_idx global. Los nodos HtoDAsync capturados copian
    // bt_stagings[layer] → d_bts[layer].
    d_bts: std.ArrayListUnmanaged(cudaz.CUdeviceptr) = .empty,
    bt_stagings: std.ArrayListUnmanaged([]c_int) = .empty,
    bt_caps: std.ArrayListUnmanaged(usize) = .empty,
    start_pos_staging: []c_int = &.{},
    seq_len_staging: []c_int = &.{},
    /// Vision (PLAN_MMPROJ 3.2): posición de CONTEXTO para el RoPE de decode
    /// separada del SLOT del KV-cache (start_pos). Con embeddings de imagen
    /// inyectados, el slot avanza n_img pero la posición del contexto sólo
    /// max(nx,ny) — sin esto, los Q de los tokens generados rotan con thetas
    /// desplazadas. Por defecto == start_pos (text-only: sin cambio).
    d_rope_pos: cudaz.CUdeviceptr = 0,
    rope_pos_staging: []c_int = &.{},
    /// Vision (PLAN_MMPROJ Fase B): pos-ids PER-TOKEN [n][4] i32 para el
    /// mropePosIdsKernel del PREFILL (tokens de imagen: t/h/w 2D). El
    /// staging se redimensiona por chunk (≤ ubatch); `d_pos_ids` device
    /// fijo. Sólo se usa con vision; text-only usa mropeKernel clásico.
    d_pos_ids: cudaz.CUdeviceptr = 0,
    pos_ids_staging: []i32 = &.{},
    pos_ids_cap: usize = 0,
    q_buf_cap: usize = 0,
    /// §5.3 STUDY: staging host PINNED del query (f16) — persistente, cero
    /// mallocs en el loop de decode (antes: alloc/free f16 por token). Lo
    /// comparten decode/prefill: se rellena, se copia, se reutiliza.
    q_h_staging: []f16 = &.{},
    /// G1 (lane-b1 2026-09-08): staging host PINNED del output (f16) del D2H
    /// de decode — mismo patrón §5.3 para q_h_staging; antes alloc/free f16
    /// por token en decode()/launchFusedKernel().
    out_h_staging: []f16 = &.{},
    /// G1 split-K (ZIG_AI_FASPLIT=1): buffer device de partials
    /// [num_q_heads][n_splits][3+head_dim] floats. Pre-allocado en
    /// ensureSplitBuffers con n_splits FIJO — graph-safe: el puntero es
    /// estable por replay (trampa de reversión decodeDevice: el grafo
    /// congela kernelParams; NUNCA reasignar tras el capture).
    d_split_partials: cudaz.CUdeviceptr = 0,
    split_n_splits: usize = 0,
    split_head_dim: usize = 0,
    split_num_q_heads: usize = 0,
    /// Escalares del launch split congelados por el grafo (punteros estables):
    /// n_splits_c/tokens_per_split_c viven aquí, NO en stack del caller.
    split_n_splits_c: c_int = 0,
    split_tokens_per_split_c: c_int = 0,

    const Self = @This();

    pub fn init(gpa: std.mem.Allocator, config: PagedConfig, stream: cudaz.CUstream) !Self {
        const cubin_path = build_options.paged_cubin;
        if (cubin_path.len == 0) return error.CudaUnavailable;
        try cudaz.ensureContext();
        const module = try cudaz.cuModuleLoad(cubin_path);
        errdefer cudaz.cuModuleUnload(module);
        var self = Self{ .allocator = gpa, .config = config, .module = module, .stream = stream };
        // Cubin extra (Lane A): best-effort — sin él, esos formatos caen a fallback.
        if (build_options.fused_extra_cubin.len > 0) {
            self.module_extra = cudaz.cuModuleLoad(build_options.fused_extra_cubin) catch |e| blk: {
                debug.dbg.printLevel(.info, "[paged_attn] fused_extra cubin no cargado ({s}); formatos extra en fallback\n", .{@errorName(e)});
                break :blk null;
            };
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.pool) |*p| p.deinit();
        if (self.paged_pool) |*p| p.deinit();
        if (self.d_q16 != 0) cudaz.cuMemFree(self.d_q16);
        if (self.d_out16 != 0) cudaz.cuMemFree(self.d_out16);
        if (self.d_seq_lens != 0) cudaz.cuMemFree(self.d_seq_lens);
        if (self.d_start_pos != 0) cudaz.cuMemFree(self.d_start_pos);
        for (self.d_bts.items) |p| {
            if (p != 0) cudaz.cuMemFree(p);
        }
        self.d_bts.deinit(self.allocator);
        for (self.bt_stagings.items) |s| {
            if (s.len > 0) cudaz.pinnedFree(c_int, s);
        }
        self.bt_stagings.deinit(self.allocator);
        self.bt_caps.deinit(self.allocator);
        if (self.start_pos_staging.len > 0) cudaz.pinnedFree(c_int, self.start_pos_staging);
        if (self.seq_len_staging.len > 0) cudaz.pinnedFree(c_int, self.seq_len_staging);
        if (self.rope_pos_staging.len > 0) cudaz.pinnedFree(c_int, self.rope_pos_staging);
        if (self.d_rope_pos != 0) cudaz.cuMemFree(self.d_rope_pos);
        if (self.pos_ids_staging.len > 0) cudaz.pinnedFree(i32, self.pos_ids_staging);
        if (self.d_pos_ids != 0) cudaz.cuMemFree(self.d_pos_ids);
        if (self.q_h_staging.len > 0) cudaz.pinnedFree(f16, self.q_h_staging);
        if (self.out_h_staging.len > 0) cudaz.pinnedFree(f16, self.out_h_staging);
        if (self.d_split_partials != 0) cudaz.cuMemFree(self.d_split_partials);
        if (self.module_extra) |m| cudaz.cuModuleUnload(m);
        cudaz.cuModuleUnload(self.module);
    }

    /// Aloca (o reutiliza) el pool de bloques del dispositivo.
    /// Prefiere el pool VMM paginado (commit físico por bloque); si el driver
    /// no soporta VMM o `block_bytes` no está alineado a granularidad, cae al
    /// pool contiguo.
    pub fn ensurePool(self: *Self, block_alloc: *BlockAllocator) !void {
        if (self.pool != null or self.paged_pool != null) return;
        if (PagedGpuBlockPool.init(
            self.allocator,
            block_alloc.numTotal(),
            block_alloc.block_bytes,
            block_alloc.quant_k,
            block_alloc.quant_v,
        )) |pp| {
            // Breadcrumb DEBUG_LEVEL>=info: modo de pool y tamaños (clave para
            // diagnosticar fallos asíncronos de kernels que indexan el pool).
            debug.dbg.printLevel(.info, "[pool] VMM paginado: {d} bloques x {d}B (k={s} v={s})\n", .{ block_alloc.numTotal(), block_alloc.block_bytes, @tagName(block_alloc.quant_k), @tagName(block_alloc.quant_v) });
            self.paged_pool = pp;
        } else |e| {
            debug.dbg.printLevel(.info, "[pool] VMM no disponible ({s}) → contiguo: {d} bloques x {d}B (k={s} v={s})\n", .{ @errorName(e), block_alloc.numTotal(), block_alloc.block_bytes, @tagName(block_alloc.quant_k), @tagName(block_alloc.quant_v) });
            self.pool = try GpuBlockPool.init(
                self.allocator,
                block_alloc.numTotal(),
                block_alloc.block_bytes,
            );
        }
    }

    pub fn cacheBase(self: *Self, block_alloc: *BlockAllocator) !cudaz.CUdeviceptr {
        try self.ensurePool(block_alloc);
        if (self.paged_pool) |*pp| return pp.vaddr;
        return self.pool.?.d_cache;
    }

    /// Marca un bloque físico como modificado en host (nuevo KV escrito). El
    /// próximo stage lo re-subirá aunque esté residente.
    pub fn markDirty(self: *Self, phys_id: usize) void {
        if (self.paged_pool) |*pp| {
            pp.markDirty(phys_id);
        } else if (self.pool) |*p| {
            p.markDirty(phys_id);
        }
    }

    fn ensureDecodeBuffers(self: *Self, q_stride: usize) !void {
        try cudaz.ensureCurrent();
        if (self.q_buf_cap < q_stride) {
            if (self.d_q16 != 0) cudaz.cuMemFree(self.d_q16);
            if (self.d_out16 != 0) cudaz.cuMemFree(self.d_out16);
            self.d_q16 = try cudaz.cuMemAlloc(q_stride * @sizeOf(f16));
            self.d_out16 = try cudaz.cuMemAlloc(q_stride * @sizeOf(f16));
            if (self.q_h_staging.len > 0) cudaz.pinnedFree(f16, self.q_h_staging);
            if (self.out_h_staging.len > 0) cudaz.pinnedFree(f16, self.out_h_staging);
            self.q_h_staging = try cudaz.pinnedAlloc(f16, q_stride);
            self.out_h_staging = try cudaz.pinnedAlloc(f16, q_stride);
            self.q_buf_cap = q_stride;
        }
        if (self.d_seq_lens == 0) self.d_seq_lens = try cudaz.cuMemAlloc(@sizeOf(c_int));
        if (self.d_start_pos == 0) self.d_start_pos = try cudaz.cuMemAlloc(@sizeOf(c_int));
    }

    /// G1 split-K: pre-alloca el buffer de partials. n_splits FIJO
    /// (SplitPolicy.forSeqLen) para que el puntero y los escalares del
    /// kernel sean estables bajo CUDA-graph replay — reasignar tras el
    /// capture congelaría punteros viejos (trampa documentada en
    /// decodeDevice: reversión histórica del capture por kernelParams).
    fn ensureSplitBuffers(self: *Self, num_q_heads: usize, head_dim: usize) !void {
        try cudaz.ensureCurrent();
        if (self.d_split_partials != 0 and
            self.split_num_q_heads == num_q_heads and self.split_head_dim == head_dim and
            self.split_n_splits == splitPolicyForSeqLen(0))
        {
            return; // forma idéntica: reutiliza (puntero estable)
        }
        const n_splits = splitPolicyForSeqLen(0);
        if (self.d_split_partials != 0) cudaz.cuMemFree(self.d_split_partials);
        const floats = num_q_heads * n_splits * (3 + head_dim);
        self.d_split_partials = try cudaz.cuMemAlloc(floats * @sizeOf(f32));
        self.split_n_splits = n_splits;
        self.split_head_dim = head_dim;
        self.split_num_q_heads = num_q_heads;
        self.split_n_splits_c = @intCast(n_splits);
        self.split_tokens_per_split_c = @intCast(splitTokensPerSplit(0));
    }

    /// Asegura los buffers device de la block table de una capa de atención
    /// (d_bt) y su staging host pinned (bt_staging). Cada capa tiene su propio
    /// par; el grafo captura nodos HtoDAsync por capa que copian
    /// bt_stagings[layer] → d_bts[layer].
    fn ensureLayerDecodeScratch(self: *Self, layer_idx: usize, max_num_blocks: usize) !void {
        try cudaz.ensureCurrent();
        while (self.d_bts.items.len <= layer_idx) {
            try self.d_bts.append(self.allocator, 0);
            try self.bt_stagings.append(self.allocator, &.{});
            try self.bt_caps.append(self.allocator, 0);
        }
        if (self.bt_caps.items[layer_idx] < max_num_blocks) {
            if (self.d_bts.items[layer_idx] != 0) cudaz.cuMemFree(self.d_bts.items[layer_idx]);
            self.d_bts.items[layer_idx] = try cudaz.cuMemAlloc(max_num_blocks * @sizeOf(c_int));
            if (self.bt_stagings.items[layer_idx].len > 0) cudaz.pinnedFree(c_int, self.bt_stagings.items[layer_idx]);
            self.bt_stagings.items[layer_idx] = try cudaz.pinnedAlloc(c_int, max_num_blocks);
            self.bt_caps.items[layer_idx] = max_num_blocks;
        }
    }

    /// Asegura el scratch de decode (staging host + buffers device) de la capa
    /// de atención `layer_idx`. En modo grafo el staging se dimensiona al
    /// presupuesto completo (budget) en el build, de modo que los punteros host
    /// capturados nunca se reasignan.
    pub fn setupDecodeScratch(self: *Self, layer_idx: usize, q_stride: usize, max_num_blocks: usize) !void {
        try self.ensureDecodeBuffers(q_stride);
        try self.ensureLayerDecodeScratch(layer_idx, max_num_blocks);
        if (self.start_pos_staging.len == 0) self.start_pos_staging = try cudaz.pinnedAlloc(c_int, 1);
        if (self.seq_len_staging.len == 0) self.seq_len_staging = try cudaz.pinnedAlloc(c_int, 1);
    }

    /// Vision (PLAN_MMPROJ 3.2): activa el buffer d_rope_pos (posición de
    /// contexto separada del slot KV para el RoPE del decode). Idempotente.
    pub fn enableRopePos(self: *Self) !void {
        try cudaz.ensureCurrent();
        if (self.d_rope_pos == 0) self.d_rope_pos = try cudaz.cuMemAlloc(@sizeOf(c_int));
        if (self.rope_pos_staging.len == 0) self.rope_pos_staging = try cudaz.pinnedAlloc(c_int, 1);
    }

    pub fn getDRopePos(self: *Self) cudaz.CUdeviceptr {
        return self.d_rope_pos;
    }

    /// Vision (PLAN_MMPROJ Fase B): activa el buffer d_pos_ids [max_n][4]
    /// para el mropePosIdsKernel del PREFIL. `max_n` = ubatch (o seq_len si
    /// el prefill no trocea). Idempotente; redimensiona si crece.
    pub fn enablePosIds(self: *Self, max_n: usize) !void {
        try cudaz.ensureCurrent();
        const need = max_n * 4;
        if (self.d_pos_ids == 0) {
            self.d_pos_ids = try cudaz.cuMemAlloc(need * @sizeOf(i32));
            self.pos_ids_staging = try cudaz.pinnedAlloc(i32, need);
            self.pos_ids_cap = need;
        } else if (need > self.pos_ids_cap) {
            // Re-alloc (raro: ubatch fija tras arranque)
            if (self.pos_ids_staging.len > 0) cudaz.pinnedFree(i32, self.pos_ids_staging);
            if (self.d_pos_ids != 0) cudaz.cuMemFree(self.d_pos_ids);
            self.d_pos_ids = try cudaz.cuMemAlloc(need * @sizeOf(i32));
            self.pos_ids_staging = try cudaz.pinnedAlloc(i32, need);
            self.pos_ids_cap = need;
        }
    }

    /// Rellena el staging de pos-ids con el chunk [start..start+n) del array
    /// host y lo sube HtoD (síncrono — prefill no está capturado por grafo).
    /// Sólo con vision activo (enablePosIds previo).
    pub fn uploadPosIds(self: *Self, ids: []const [4]i32, start: usize, n: usize) !void {
        if (self.d_pos_ids == 0) return; // no vision
        const cap_n = self.pos_ids_cap / 4;
        if (n > cap_n) return error.PosIdsChunkTooBig;
        for (0..n) |k| {
            const src = ids[start + k];
            self.pos_ids_staging[k * 4 + 0] = src[0];
            self.pos_ids_staging[k * 4 + 1] = src[1];
            self.pos_ids_staging[k * 4 + 2] = src[2];
            self.pos_ids_staging[k * 4 + 3] = src[3];
        }
        try cudaz.cuMemcpyHtoD(self.d_pos_ids, @intFromPtr(self.pos_ids_staging.ptr), n * 4 * @sizeOf(i32));
    }

    pub fn getDPosIds(self: *Self) cudaz.CUdeviceptr {
        return self.d_pos_ids;
    }

    /// Rellena el staging host del decode de la capa `layer_idx` (block table +
    /// start_pos + seq_len + rope_pos). No toca el device: solo pinta el
    /// staging que el grafo copia. `rope_pos` (opcional): posición de CONTEXTO
    /// para el RoPE cuando difiere del slot KV (vision: embeddings de imagen
    /// consumen n_img slots pero max(nx,ny) posiciones). null ⇒ start_pos.
    pub fn fillStaging(self: *Self, layer_idx: usize, block_table: *const BlockTable, start_pos: usize, seq_len: usize, rope_pos: ?usize) void {
        self.start_pos_staging[0] = @intCast(start_pos);
        self.seq_len_staging[0] = @intCast(seq_len);
        if (self.rope_pos_staging.len > 0) {
            self.rope_pos_staging[0] = @intCast(rope_pos orelse start_pos);
        }
        // G1c fix (lane-c 2026-09-12): la cobertura dinámica vive IN-KERNEL
        // (ceil(seq_len/n_splits) del seq_lens DEVICE — CUDA-graph-safe:
        // un escalar host en kernelParams queda CONGELADO en el ejecutable
        // capturado; el valor por-token solo puede venir de memoria device).
        // Aquí no queda nada que actualizar.
        for (self.bt_stagings.items[layer_idx], 0..) |*d, i| {
            d.* = if (block_table.getPhysical(i)) |phys| @intCast(phys) else -1;
        }
    }

    /// HtoDAsync del staging host de la capa `layer_idx` (bloqueable/grafo):
    /// d_bt[layer], d_start_pos, d_seq_lens, d_rope_pos. Ordenado en el stream.
    pub fn uploadScratch(self: *Self, layer_idx: usize) !void {
        try cudaz.ensureCurrent();
        try cudaz.cuMemcpyHtoDAsync(self.d_bts.items[layer_idx], @intFromPtr(self.bt_stagings.items[layer_idx].ptr), self.bt_stagings.items[layer_idx].len * @sizeOf(c_int), self.stream);
        try cudaz.cuMemcpyHtoDAsync(self.d_start_pos, @intFromPtr(self.start_pos_staging.ptr), @sizeOf(c_int), self.stream);
        try cudaz.cuMemcpyHtoDAsync(self.d_seq_lens, @intFromPtr(self.seq_len_staging.ptr), @sizeOf(c_int), self.stream);
        if (self.d_rope_pos != 0 and self.rope_pos_staging.len > 0) {
            try cudaz.cuMemcpyHtoDAsync(self.d_rope_pos, @intFromPtr(self.rope_pos_staging.ptr), @sizeOf(c_int), self.stream);
        }
    }

    /// HtoDAsync del start_pos solo (path de prefill, no capturado).
    pub fn uploadStartPos(self: *Self, start_pos: usize) !void {
        try cudaz.ensureCurrent();
        try self.ensureDecodeBuffers(0);
        if (self.start_pos_staging.len == 0) self.start_pos_staging = try cudaz.pinnedAlloc(c_int, 1);
        self.start_pos_staging[0] = @intCast(start_pos);
        try cudaz.cuMemcpyHtoDAsync(self.d_start_pos, @intFromPtr(self.start_pos_staging.ptr), @sizeOf(c_int), self.stream);
    }

    pub fn getDStartPos(self: *Self) cudaz.CUdeviceptr {
        return self.d_start_pos;
    }

    pub fn getDSeqLens(self: *Self) cudaz.CUdeviceptr {
        return self.d_seq_lens;
    }

    fn stageBlocks(self: *Self, block_alloc: *BlockAllocator, block_table: *const BlockTable) !void {
        debug.dbg.printLevel(.detail, "[gpu_kernels] stageBlocks called\n", .{});
        try self.ensurePool(block_alloc);
        debug.dbg.printLevel(.detail, "[gpu_kernels] ensurePool done\n", .{});
        if (self.paged_pool) |*pp| {
            debug.dbg.printLevel(.detail, "[gpu_kernels] using paged_pool\n", .{});
            try pp.stageTable(block_alloc, block_table);
            debug.dbg.printLevel(.detail, "[gpu_kernels] paged_pool.stageTable done\n", .{});
        } else {
            debug.dbg.printLevel(.detail, "[gpu_kernels] using pool\n", .{});
            try self.pool.?.stageTable(block_alloc, block_table);
            debug.dbg.printLevel(.detail, "[gpu_kernels] pool.stageTable done\n", .{});
        }
    }

    /// Sube todos los bloques de una block table al device (uso puntual tras el
    /// prefill CPU, antes del decode GPU-residente).
    pub fn stageTableAll(self: *Self, block_alloc: *BlockAllocator, block_table: *const BlockTable) !void {
        try self.stageBlocks(block_alloc, @constCast(block_table));
    }

    fn stageBlock(self: *Self, block_alloc: *BlockAllocator, phys_id: usize) !void {
        try self.ensurePool(block_alloc);
        if (self.paged_pool) |*pp| {
            // Use quantized staging for quantized KV formats
            if (block_alloc.quant_k.hasScales() or block_alloc.quant_v.hasScales()) {
                try pp.stageQuantBlock(block_alloc, phys_id);
            } else {
                try pp.stageBlock(block_alloc, phys_id);
            }
        } else {
            try self.pool.?.stageBlock(block_alloc, phys_id);
        }
    }

    fn evictBlocks(self: *Self, block_alloc: *BlockAllocator, phys_ids: []const usize) !void {
        if (self.paged_pool) |*pp| {
            try pp.evictBlocks(block_alloc, phys_ids);
        } else if (self.pool) |*p| {
            try p.evictBlocks(block_alloc, phys_ids);
        }
    }

    fn evictBlock(self: *Self, block_alloc: *BlockAllocator, phys_id: usize) !void {
        if (self.paged_pool) |*pp| {
            try pp.evictBlock(block_alloc, phys_id);
        } else if (self.pool) |*p| {
            try p.evictBlock(block_alloc, phys_id);
        }
    }

    /// Decode de un token (1 secuencia). `query`/`out` en f32; K/V se leen del
    /// memory-pool de `block_alloc` como f16. Equivale a `PagedAttention.decode`.
    pub fn decode(
        self: *Self,
        query: []const f32,
        out: []f32,
        block_table: BlockTable,
        block_alloc: *BlockAllocator,
        config: PagedConfig,
    ) !void {
        const num_q_heads = config.num_q_heads;
        const num_kv_heads = config.num_kv_heads;
        const head_dim = config.head_dim;
        const block_size = config.block_size;
        const q_stride = num_q_heads * head_dim;
        const seq_len = block_table.num_tokens;

        std.debug.assert(query.len == q_stride);
        std.debug.assert(out.len == q_stride);
        const max_num_blocks = block_table.numBlocks();

        try cudaz.ensureCurrent();

        // §5.3 STUDY: staging persistente (pinned) — cero allocs por token.
        try self.ensureDecodeBuffers(q_stride);
        const q_f16 = self.q_h_staging;
        for (query, 0..) |v, i| q_f16[i] = @floatCast(v);

        // §5.3 STUDY: staging de block table persistente por capa (reutiliza
        // el pinned buffer del grafo; antes: alloc/free c_int por token).
        try self.ensureLayerDecodeScratch(0, max_num_blocks);
        const bt_host = self.bt_stagings.items[0];
        for (0..max_num_blocks) |i| {
            bt_host[i] = if (block_table.getPhysical(i)) |phys| @intCast(phys) else -1;
        }
        // Subir solo los bloques referenciados por la block table.
        debug.dbg.printLevel(.detail, "[decode] calling stageBlocks\n", .{});
        try self.stageBlocks(block_alloc, &block_table);
        debug.dbg.printLevel(.detail, "[decode] stageBlocks done\n", .{});

        debug.dbg.printLevel(.detail, "[decode] ensureDecodeBuffers\n", .{});
        debug.dbg.printLevel(.detail, "[decode] ensureDecodeBuffers done\n", .{});

        debug.dbg.printLevel(.detail, "[decode] ensureLayerDecodeScratch done\n", .{});
        var d_bt = self.d_bts.items[0];

        var seq_len_c: c_int = @intCast(seq_len);

        debug.dbg.printLevel(.detail, "[decode] cuMemcpyHtoD q16\n", .{});
        try cudaz.cuMemcpyHtoD(self.d_q16, @intFromPtr(q_f16.ptr), q_stride * @sizeOf(f16));
        debug.dbg.printLevel(.detail, "[decode] cuMemcpyHtoD q16 done\n", .{});
        debug.dbg.printLevel(.detail, "[decode] cuMemcpyHtoD d_bt\n", .{});
        try cudaz.cuMemcpyHtoD(d_bt, @intFromPtr(bt_host.ptr), max_num_blocks * @sizeOf(c_int));
        debug.dbg.printLevel(.detail, "[decode] cuMemcpyHtoD d_bt done\n", .{});
        debug.dbg.printLevel(.detail, "[decode] cuMemcpyHtoD seq_lens\n", .{});
        try cudaz.cuMemcpyHtoD(self.d_seq_lens, @intFromPtr(&seq_len_c), @sizeOf(c_int));
        debug.dbg.printLevel(.detail, "[decode] cuMemcpyHtoD seq_lens done\n", .{});

        // Dispatch using kernel dispatch table. Formats without a fused kernel
        // in the cubin (KernelNotFound) fall back to the fp16 path below instead
        // of erroring — the dispatch table registers the full target set while
        // kernels land incrementally.
        const dispatch_map = try getKernelDispatch(self.allocator);
        const k_fmt = config.quant_k;
        const v_fmt = config.quant_v;

        if (k_fmt != .fp16 or v_fmt != .fp16) {
            if (dispatch_map.get(k_fmt)) |k_dispatch| {
                if (dispatch_map.get(v_fmt)) |v_dispatch| {
                    if (k_dispatch.super_block == v_dispatch.super_block and k_fmt == v_fmt) {
                        const fused_res = self.launchFusedKernel(k_dispatch, query, out, &block_table, block_alloc, config);
                        if (fused_res) {
                            return;
                        } else |err| switch (err) {
                            error.KernelNotFound => {
                                debug.dbg.printLevel(.info, "[paged_attn] fused {s} kernel missing en cubin; fallback fp16\n", .{@tagName(k_fmt)});
                            },
                            else => return err,
                        }
                    }
                }
            }
        }

        // Fallback: fp16 attention kernel over staged blocks.
        const func = cudaz.cuModuleGetFunction(self.module, "paged_attention_decode_f16_kernel") catch return error.KernelNotFound;

        debug.dbg.printLevel(.detail, "[decode] kernel found, preparing launch\n", .{});

        // Scalar params: kernel recibe int por valor; el driver los lee del host.
        var num_seqs_c: c_int = 1;
        var max_blocks_c: c_int = @intCast(max_num_blocks);
        var num_q_c: c_int = @intCast(num_q_heads);
        var num_kv_c: c_int = @intCast(num_kv_heads);
        var head_dim_c: c_int = @intCast(head_dim);
        var block_size_c: c_int = @intCast(block_size);

        var d_cache_v = try self.cacheBase(block_alloc);

        var kp: [11]?*anyopaque = .{
            &self.d_out16,    &self.d_q16, &d_cache_v,    &d_bt,
            &self.d_seq_lens, &num_seqs_c, &max_blocks_c, &num_q_c,
            &num_kv_c,        &head_dim_c, &block_size_c,
        };
        const shared_bytes: c_uint = @intCast(2 * head_dim * @sizeOf(f32));
        debug.dbg.printLevel(.detail, "[decode] cuLaunchKernel grid=(1,{},1) block=(32,1,1) shared={}\n", .{ num_q_heads, shared_bytes });
        try cudaz.cuLaunchKernel(func, 1, @intCast(num_q_heads), 1, 32, 1, 1, shared_bytes, self.stream, @ptrCast(&kp), null);
        debug.dbg.printLevel(.detail, "[decode] cuLaunchKernel done\n", .{});
        debug.dbg.printLevel(.detail, "[decode] cuStreamSynchronize\n", .{});
        try cudaz.cuStreamSynchronize(self.stream);
        debug.dbg.printLevel(.detail, "[decode] cuStreamSynchronize done\n", .{});

        // G1 (lane-b1): D2H sobre staging pinned persistente (§5.3) —
        // antes alloc/free f16 por token.
        const out_f16 = self.out_h_staging;
        debug.dbg.printLevel(.detail, "[decode] cuMemcpyDtoH\n", .{});
        try cudaz.cuMemcpyDtoH(@intFromPtr(out_f16.ptr), self.d_out16, q_stride * @sizeOf(f16));
        debug.dbg.printLevel(.detail, "[decode] cuMemcpyDtoH done\n", .{});

        for (out_f16, 0..) |v, i| out[i] = @floatCast(v);
    }

    /// Unified launcher for all fused quantized kernels.
    /// Uses the dispatch table entry to determine kernel parameters.
    /// Busca un kernel fusionado en el cubin principal y luego en el extra
    /// (Lane A). Permite añadir formatos sin tocar paged_attention.cu.
    fn getFusedFunc(self: *Self, name: []const u8) !cudaz.CUfunction {
        if (cudaz.cuModuleGetFunction(self.module, name)) |f| {
            return f;
        } else |_| {}
        if (self.module_extra) |m| {
            if (cudaz.cuModuleGetFunction(m, name)) |f| {
                debug.dbg.printLevel(.detail, "[paged_attn] {s} resuelto en cubin extra\n", .{name});
                return f;
            } else |_| {}
        }
        return error.KernelNotFound;
    }

    fn launchFusedKernel(
        self: *Self,
        dispatch: KernelDispatch,
        query: []const f32,
        out: []f32,
        block_table: *const BlockTable,
        block_alloc: *BlockAllocator,
        config: PagedConfig,
    ) !void {
        const num_q_heads = config.num_q_heads;
        const num_kv_heads = config.num_kv_heads;
        const head_dim = config.head_dim;
        const block_size = config.block_size;
        const q_stride = num_q_heads * head_dim;
        const seq_len = block_table.num_tokens;

        std.debug.assert(query.len == q_stride);
        std.debug.assert(out.len == q_stride);

        try cudaz.ensureCurrent();

        // Stage quantized blocks (data + scales)
        try self.stageBlocks(block_alloc, @constCast(block_table));

        // §5.3 STUDY: staging persistente — cero allocs por llamada.
        try self.ensureDecodeBuffers(q_stride);
        const q_f16 = self.q_h_staging;
        for (query, 0..) |v, i| q_f16[i] = @floatCast(v);

        try self.ensureLayerDecodeScratch(0, block_table.numBlocks());
        const bt_host = self.bt_stagings.items[0];
        for (0..block_table.numBlocks()) |i| {
            bt_host[i] = if (block_table.getPhysical(i)) |phys| @intCast(phys) else -1;
        }
        var d_bt = self.d_bts.items[0];

        var seq_len_c: c_int = @intCast(seq_len);

        try cudaz.cuMemcpyHtoD(self.d_q16, @intFromPtr(q_f16.ptr), q_stride * @sizeOf(f16));
        try cudaz.cuMemcpyHtoD(d_bt, @intFromPtr(bt_host.ptr), block_table.numBlocks() * @sizeOf(c_int));
        try cudaz.cuMemcpyHtoD(self.d_seq_lens, @intFromPtr(&seq_len_c), @sizeOf(c_int));

        // Get GPU pointers for quantized cache and scales.
        const d_cache_kv = try self.cacheBase(block_alloc);
        // Los kernels con escala EMBEBIDA en el bloque (q8_0 reescrito,
        // K-quants fusionados) ignoran estos punteros; si el pool cayó al
        // contiguo (sin VMM) no hay arrays de escalas y se pasa null — no es
        // error. Sólo los kernels legacy de escalas separadas los usarían.
        const d_k_scales: usize = if (self.paged_pool) |*pp| pp.d_k_scales else 0;
        const d_v_scales: usize = if (self.paged_pool) |*pp| pp.d_v_scales else 0;

        const func = self.getFusedFunc(dispatch.kernel_name) catch return error.KernelNotFound;

        // Common kernel parameters
        var num_seqs_c: c_int = 1;
        var max_blocks_c: c_int = @intCast(block_table.numBlocks());
        var num_q_c: c_int = @intCast(num_q_heads);
        var num_kv_c: c_int = @intCast(num_kv_heads);
        var head_dim_c: c_int = @intCast(head_dim);
        var block_size_c: c_int = @intCast(block_size);

        // For super-block formats, we need scales; for 32-element formats we don't
        var kp: [13]?*anyopaque = undefined;
        kp[0] = &self.d_out16;
        kp[1] = &self.d_q16;

        if (dispatch.super_block) {
            kp[2] = @ptrCast(@constCast(&d_cache_kv));
            kp[3] = @as(?*anyopaque, @ptrFromInt(@intFromPtr(&d_k_scales)));
            kp[4] = @as(?*anyopaque, @ptrFromInt(@intFromPtr(&d_v_scales)));
            kp[5] = &d_bt;
            kp[6] = &self.d_seq_lens;
            kp[7] = &num_seqs_c;
            kp[8] = &max_blocks_c;
            kp[9] = &num_q_c;
            kp[10] = &num_kv_c;
            kp[11] = &head_dim_c;
            kp[12] = &block_size_c;

            // Shared memory: accumulators + scale cache
            const scale_blocks = (config.block_size * config.num_kv_heads * config.head_dim + 255) / 256;
            const shared_bytes: c_uint = @intCast(2 * head_dim * @sizeOf(f32) + 2 * scale_blocks * @sizeOf(f32));
            try cudaz.cuLaunchKernel(func, 1, @intCast(num_q_heads), 1, 32, 1, 1, shared_bytes, self.stream, @ptrCast(&kp), null);
        } else {
            // 32-element block formats (q8_0, q4_0, etc.)
            kp[2] = @ptrCast(@constCast(&d_cache_kv));
            kp[3] = @as(?*anyopaque, @ptrFromInt(@intFromPtr(&d_k_scales)));
            kp[4] = @as(?*anyopaque, @ptrFromInt(@intFromPtr(&d_v_scales)));
            kp[5] = &d_bt;
            kp[6] = &self.d_seq_lens;
            kp[7] = &num_seqs_c;
            kp[8] = &max_blocks_c;
            kp[9] = &num_q_c;
            kp[10] = &num_kv_c;
            kp[11] = &head_dim_c;
            kp[12] = &block_size_c;

            const shared_bytes: c_uint = @intCast(2 * head_dim * @sizeOf(f32) + 2 * ((config.block_size * config.num_kv_heads * config.head_dim + 31) / 32) * @sizeOf(f32));
            try cudaz.cuLaunchKernel(func, 1, @intCast(num_q_heads), 1, 32, 1, 1, shared_bytes, self.stream, @ptrCast(&kp), null);
        }

        try cudaz.cuStreamSynchronize(self.stream);

        // G1 (lane-b1): D2H sobre staging pinned persistente (§5.3).
        const out_f16 = self.out_h_staging;
        try cudaz.cuMemcpyDtoH(@intFromPtr(out_f16.ptr), self.d_out16, q_stride * @sizeOf(f16));

        for (out_f16, 0..) |v, i| out[i] = @floatCast(v);
    }

    // G1 (lane-b1 2026-09-08): decodeQuantized/decodeQ8_0/decodeQ4_0/decodeQ4_K
    // ELIMINADAS — código muerto (cero call-sites, verificado por grep en
    // src/+tests/): duplicaban el patrón alloc/H2D/sync/D2H ya unificado en
    // decode()+launchFusedKernel(); decodeQ8_0 incluso tenía un doble bucle
    // de copy pegado (bug cosmético). El dispatch vivo es la tabla de
    // getKernelDispatch() + launchFusedKernel().

    // ─── Ruta decode 100% device (Phase 1b) ─────────────────────────────────
    // q16/out16 ya viven en GPU (AttentionGpu); el KV se escribe con
    // kvAppendF16 y el decode lee el pool sin staging ni sync. El llamador
    // sincroniza el stream una vez por token.
    pub fn uploadBlockTable(self: *Self, layer_idx: usize, bt_host: []const c_int) !void {
        try cudaz.ensureCurrent();
        try self.ensureDecodeBuffers(self.config.num_q_heads * self.config.head_dim);
        try self.ensureLayerDecodeScratch(layer_idx, bt_host.len);
        try cudaz.cuMemcpyHtoDAsync(self.d_bts.items[layer_idx], @intFromPtr(bt_host.ptr), bt_host.len * @sizeOf(c_int), self.stream);
    }

    pub fn getDbt(self: *Self, layer_idx: usize) cudaz.CUdeviceptr {
        return self.d_bts.items[layer_idx];
    }

    /// El bloque será escrito por GPU (kvAppendF16) en lugar de H2D: lo marca
    /// residente/committed sin copiar para que el decode device pueda leerlo.
    pub fn ensureBlockCommitted(self: *Self, block_alloc: *BlockAllocator, phys_id: usize) !void {
        try self.ensurePool(block_alloc);
        if (self.paged_pool) |*pp| {
            try pp.ensureCommitted(phys_id);
        } else if (self.pool) |*p| {
            p.ensureCommitted(phys_id);
        }
    }

    /// D2H async (stream-ordered) del bloque escrito por GPU → host pool
    /// (mantiene el pool host autoritativo para COW/scheduler).
    pub fn syncBlockToHost(self: *Self, block_alloc: *BlockAllocator, phys_id: usize) !void {
        if (self.paged_pool) |*pp| {
            try pp.syncBlockToHost(block_alloc, phys_id, self.stream);
        } else if (self.pool) |*p| {
            try p.syncBlockToHost(block_alloc, phys_id, self.stream);
        }
    }

    /// Decode device→device: `q16`/`out16` son buffers f16 de GPU. La block
    /// table, start_pos y seq_len ya están en device (subidos por
    /// `uploadScratch`, o por los nodos HtoDAsync capturados del grafo de
    /// decode). Sin staging de bloques y sin sync.
    pub fn decodeDevice(
        self: *Self,
        layer_idx: usize,
        q16: cudaz.CUdeviceptr,
        out16: cudaz.CUdeviceptr,
        block_alloc: *BlockAllocator,
    ) !void {
        const config = self.config;
        const num_q_heads = config.num_q_heads;
        const num_kv_heads = config.num_kv_heads;
        const head_dim = config.head_dim;
        const block_size = config.block_size;
        const max_num_blocks = self.bt_stagings.items[layer_idx].len;

        try cudaz.ensureCurrent();

        // Selección de kernel por formato KV. El path device→device exige
        // quant_k == quant_v (mixto caería a dos pasadas, no soportado aquí).
        const elems_per_block = block_size * num_kv_heads * head_dim;
        const is_quant = config.quant_k != .fp16 or config.quant_v != .fp16;
        if (is_quant and config.quant_k != config.quant_v) return error.KvQuantUnsupported;

        // G1 split-K (lane-b1): variante flash-decoding f16 opt-in
        // ZIG_AI_FASPLIT=1 — grid (1,H,n_splits)×128 vs (1,H,1)×32 del
        // base: con GQA el base deja 84 SMs casi vacíos (8 warps en 0.8B).
        // Partial+combine; escalares/punteros ESTABLES (graph-safe, ver
        // ensureSplitBuffers). Default OFF hasta paridad greedy + ≥1.5×.
        if (fasplitEnabled() and !is_quant) {
            return self.decodeDeviceSplit(layer_idx, q16, out16, block_alloc);
        }
        // Lane A C5 (request lane-c): los formatos del cubin extra también
        // tienen camino device→device. Misma firma de 13 params (Contrato C1:
        // escalas embebidas, k/v_scales ignorados por el kernel) y mismo smem
        // base (2*hd floats, extra_smem_floats=0).
        var extra_smem_floats: usize = 0;
        const func: cudaz.CUfunction = blk: {
            if (deviceDecodeKernel(config.quant_k, elems_per_block, head_dim)) |inf| {
                extra_smem_floats = inf.extra_smem_floats;
                break :blk cudaz.cuModuleGetFunction(self.module, inf.name) catch
                    return error.KernelNotFound;
            }
            const extra_name: []const u8 = switch (config.quant_k) {
                .iq4_xs => "paged_attention_decode_iq4_xs_kernel",
                // T1-B2 (request lane-c 21:05 / B 11:15): append bit-exacto
                // (B2.5/B2.7) + prefill causal universal verdes; mismo patrón
                // iq4_xs (13 params, escalas embebidas, smem base).
                .iq1_s => "paged_attention_decode_iq1_s_kernel",
                .iq1_m => "paged_attention_decode_iq1_m_kernel",
                .iq3_s => "paged_attention_decode_iq3_s_kernel",
                // Ticket 7.5 cierre (coordinador): los kernels de decode
                // existían en el cubin extra (fused_decode_extra.cu:627-630)
                // pero NO estaban cableados aquí ⇒ KvQuantUnsupported al
                // reactivar el gate. Con esto el pipeline iq2 es completo
                // (append + prefill + decode) y el 7.5 queda CERRADO.
                .iq2_s => "paged_attention_decode_iq2_s_kernel",
                .iq2_xs => "paged_attention_decode_iq2_xs_kernel",
                .iq2_xxs => "paged_attention_decode_iq2_xxs_kernel",
                .iq3_xxs => "paged_attention_decode_iq3_xxs_kernel",
                .iq4_nl => "paged_attention_decode_iq4_nl_kernel",
                .q4_1 => "paged_attention_decode_q4_1_kernel",
                .q5_0 => "paged_attention_decode_q5_0_kernel",
                .q5_1 => "paged_attention_decode_q5_1_kernel",
                .q8_1 => "paged_attention_decode_q8_1_kernel",
                .tq1_0 => "paged_attention_decode_tq1_0_kernel",
                .tq2_0 => "paged_attention_decode_tq2_0_kernel",
                .mxfp4 => "paged_attention_decode_mxfp4_kernel",
                else => return error.KvQuantUnsupported,
            };
            const m = self.module_extra orelse return error.KernelNotFound;
            break :blk cudaz.cuModuleGetFunction(m, extra_name) catch
                return error.KernelNotFound;
        };

        g_decode_persistent.num_seqs_c = 1;
        g_decode_persistent.max_blocks_c = @intCast(max_num_blocks);
        g_decode_persistent.num_q_c = @intCast(num_q_heads);
        g_decode_persistent.num_kv_c = @intCast(num_kv_heads);
        g_decode_persistent.head_dim_c = @intCast(head_dim);
        g_decode_persistent.block_size_c = @intCast(block_size);

        g_decode_persistent.d_cache_v = try self.cacheBase(block_alloc);
        g_decode_persistent.q16v = q16;
        g_decode_persistent.out16v = out16;

        if (is_quant) {
            // Escala embebida: los kernels fusionados ignoran k/v_scales; con
            // pool contiguo (fallback sin VMM) no hay arrays y se pasa null.
            const pp = self.paged_pool;
            g_decode_persistent.d_k_scales_v = if (pp) |*p| p.d_k_scales else 0;
            g_decode_persistent.d_v_scales_v = if (pp) |*p| p.d_v_scales else 0;
            g_decode_persistent.kp = [_]?*anyopaque{
                &g_decode_persistent.out16v,       &g_decode_persistent.q16v,
                &g_decode_persistent.d_cache_v,    &g_decode_persistent.d_k_scales_v,
                &g_decode_persistent.d_v_scales_v, &self.d_bts.items[layer_idx],
                &self.d_seq_lens,                  &g_decode_persistent.num_seqs_c,
                &g_decode_persistent.max_blocks_c, &g_decode_persistent.num_q_c,
                &g_decode_persistent.num_kv_c,     &g_decode_persistent.head_dim_c,
                &g_decode_persistent.block_size_c,
            };
        } else {
            g_decode_persistent.kp = [_]?*anyopaque{
                &g_decode_persistent.out16v,   &g_decode_persistent.q16v,       &g_decode_persistent.d_cache_v,    &self.d_bts.items[layer_idx],
                &self.d_seq_lens,              &g_decode_persistent.num_seqs_c, &g_decode_persistent.max_blocks_c, &g_decode_persistent.num_q_c,
                &g_decode_persistent.num_kv_c, &g_decode_persistent.head_dim_c, &g_decode_persistent.block_size_c, undefined,
                undefined,
            };
        }
        const shared_bytes: c_uint = @intCast((2 * head_dim + extra_smem_floats) * @sizeOf(f32));
        debug.dbg.printLevel(.detail, "[decode_device] launch grid=(1,{d},1) block=32 smem={d}B\n", .{ num_q_heads, shared_bytes });
        try cudaz.cuLaunchKernel(func, 1, @intCast(num_q_heads), 1, 32, 1, 1, shared_bytes, self.stream, @ptrCast(&g_decode_persistent.kp), null);
        // NOTA lane-b: un intento de CUDA Graph capture aquí fue REVERTIDO
        // (el grafo congela kernelParams; replay con q16/out16/bt distintos
        // usaría punteros viejos ⇒ corrupción silenciosa). Ver HANDOFFS
        // 23:59 B3-v2. Si se reintenta, usar cudaGraphExecKernelNodeSetParams
        // por llamada o punteros estables por forma.
    }

    /// G1 (lane-b1 2026-09-08): decode f16 flash-decoding split-K,
    /// ZIG_AI_FASPLIT=1. Dos launches: partial grid (1,H,n_splits)×128
    /// (online-softmax por chunk de tokens) + combine grid (H,)×128.
    /// Los escalares y el buffer de partials viven en `self` (estables
    /// bajo capture/replay) — MISMA trampa que arriba: nunca alocar ni
    /// mutar punteros aquí que el grafo haya congelado; ensureSplitBuffers
    /// se llama idempotente y NO reasigna si la forma no cambia.
    fn decodeDeviceSplit(
        self: *Self,
        layer_idx: usize,
        q16: cudaz.CUdeviceptr,
        out16: cudaz.CUdeviceptr,
        block_alloc: *BlockAllocator,
    ) !void {
        const config = self.config;
        const num_q_heads = config.num_q_heads;
        const num_kv_heads = config.num_kv_heads;
        const head_dim = config.head_dim;
        const block_size = config.block_size;
        const max_num_blocks = self.bt_stagings.items[layer_idx].len;

        // ensureSplitBuffers PRIMERO: inicializa split_n_splits (y el buffer
        // device de partials). Leer n_splits antes devolvería 0 en la primera
        // llamada ⇒ grid.z=0 ⇒ CUDA_ERROR_INVALID_VALUE en el launch.
        try self.ensureSplitBuffers(num_q_heads, head_dim);
        const n_splits = self.split_n_splits;

        try cudaz.ensureCurrent();

        const d_cache_v = try self.cacheBase(block_alloc);

        const func_partial = try cudaz.cuModuleGetFunction(self.module, "paged_attention_decode_f16_split_kernel");
        const func_combine = try cudaz.cuModuleGetFunction(self.module, "paged_attention_decode_f16_split_combine_kernel");

        // Escalares con vida estable (self/g_decode_persistent, no stack)
        // para kernelParams — el grafo congela las DIRECCIONES y relee los
        // valores en cada replay (trampa de la reversión histórica).
        g_decode_persistent.split_partials_v = self.d_split_partials;
        g_decode_persistent.q16v = q16;
        g_decode_persistent.out16v = out16;
        g_decode_persistent.d_cache_v = d_cache_v;
        g_decode_persistent.num_seqs_c = 1;
        g_decode_persistent.max_blocks_c = @intCast(max_num_blocks);
        g_decode_persistent.num_q_c = @intCast(num_q_heads);
        g_decode_persistent.num_kv_c = @intCast(num_kv_heads);
        g_decode_persistent.head_dim_c = @intCast(head_dim);
        g_decode_persistent.block_size_c = @intCast(block_size);
        self.split_n_splits_c = @intCast(n_splits);
        // G1c: el param host queda como NOMINAL (128) — el kernel deriva el
        // efectivo del seq_lens device (cobertura total, graph-safe).
        self.split_tokens_per_split_c = @intCast(splitTokensPerSplit(0));

        // ── Launch 1: partial. FIRMA 13 params: (partials, query, cache,
        // block_tables, seq_lens, num_seqs, max_blocks, num_q, num_kv,
        // head_dim, block_size, n_splits, tokens_per_split).
        g_decode_persistent.kp_split = [_]?*anyopaque{
            &g_decode_persistent.split_partials_v,
            &g_decode_persistent.q16v,
            &g_decode_persistent.d_cache_v,
            &self.d_bts.items[layer_idx],
            &self.d_seq_lens,
            &g_decode_persistent.num_seqs_c,
            &g_decode_persistent.max_blocks_c,
            &g_decode_persistent.num_q_c,
            &g_decode_persistent.num_kv_c,
            &g_decode_persistent.head_dim_c,
            &g_decode_persistent.block_size_c,
            &self.split_n_splits_c,
            &self.split_tokens_per_split_c,
        };
        const shared_bytes: c_uint = @intCast(2 * head_dim * @sizeOf(f32));
        debug.dbg.printLevel(.detail, "[decode_device_split] partial grid=(1,{d},{d}) block=128 smem={d}B\n", .{ num_q_heads, n_splits, shared_bytes });
        try cudaz.cuLaunchKernel(func_partial, 1, @intCast(num_q_heads), @intCast(n_splits), 128, 1, 1, shared_bytes, self.stream, @ptrCast(&g_decode_persistent.kp_split), null);

        // ── Launch 2: combine. (out, partials, num_q, head_dim, n_splits).
        g_decode_persistent.kp_split_combine = [_]?*anyopaque{
            &g_decode_persistent.out16v,
            &g_decode_persistent.split_partials_v,
            &g_decode_persistent.num_q_c,
            &g_decode_persistent.head_dim_c,
            &self.split_n_splits_c,
        };
        try cudaz.cuLaunchKernel(func_combine, @intCast(num_q_heads), 1, 1, 128, 1, 1, 0, self.stream, @ptrCast(&g_decode_persistent.kp_split_combine), null);
    }

    /// Prefill device→device causal (chunks): `q16`/`out16` son buffers f16 de
    /// GPU con `n_queries` tokens; atienden causalmente sobre los tokens ya
    /// escritos en el pool (posiciones [0..start_pos+n_queries)). `start_pos`
    /// es la posición absoluta del primer query. Sin staging de bloques ni sync.
    pub fn prefillDevice(
        self: *Self,
        layer_idx: usize,
        q16: cudaz.CUdeviceptr,
        out16: cudaz.CUdeviceptr,
        block_alloc: *BlockAllocator,
        bt_host: []const c_int,
        n_queries: usize,
        start_pos: usize,
        bt_override: ?cudaz.CUdeviceptr,
    ) !void {
        return self.prefillDeviceEx(layer_idx, q16, out16, block_alloc, bt_host, n_queries, start_pos, bt_override, true);
    }

    /// A5 (request lane-c 20:25): variante con máscara parametrizable.
    /// causal=false → atención bidireccional en [0..start_pos+n_queries)
    /// para TODAS las filas (denoiser DFlash: [id_last, MASK×bs-1]).
    pub fn prefillDeviceEx(
        self: *Self,
        layer_idx: usize,
        q16: cudaz.CUdeviceptr,
        out16: cudaz.CUdeviceptr,
        block_alloc: *BlockAllocator,
        bt_host: []const c_int,
        n_queries: usize,
        start_pos: usize,
        bt_override: ?cudaz.CUdeviceptr,
        causal: bool,
    ) !void {
        _ = layer_idx;
        const config = self.config;

        if (config.quant_v != config.quant_k) return error.KvQuantUnsupported;
        const main_kernel: ?[]const u8 = switch (config.quant_k) {
            .fp16 => "paged_attention_prefill_f16_kernel",
            .q8_0 => "paged_attention_prefill_q8_0_kernel",
            else => null,
        };
        // Lane A: cobertura universal de prefill via cubin extra (valfns
        // bit-exacto validadas por harness). fp16/q8_0 viven en el principal.
        const extra_kernel: ?[]const u8 = switch (config.quant_k) {
            .q4_0 => "paged_attention_prefill_q4_0_kernel",
            .iq4_xs => "paged_attention_prefill_iq4_xs_kernel",
            .q8_k => "paged_attention_prefill_q8_k_kernel",
            .q4_1 => "paged_attention_prefill_q4_1_kernel",
            .q5_0 => "paged_attention_prefill_q5_0_kernel",
            .q5_1 => "paged_attention_prefill_q5_1_kernel",
            .q8_1 => "paged_attention_prefill_q8_1_kernel",
            .iq3_s => "paged_attention_prefill_iq3_s_kernel",
            .iq1_s => "paged_attention_prefill_iq1_s_kernel",
            .iq1_m => "paged_attention_prefill_iq1_m_kernel",
            .tq1_0 => "paged_attention_prefill_tq1_0_kernel",
            .tq2_0 => "paged_attention_prefill_tq2_0_kernel",
            .mxfp4 => "paged_attention_prefill_mxfp4_kernel",
            .iq4_nl => "paged_attention_prefill_iq4_nl_kernel",
            .iq2_xxs => "paged_attention_prefill_iq2_xxs_kernel",
            .iq2_xs => "paged_attention_prefill_iq2_xs_kernel",
            .iq2_s => "paged_attention_prefill_iq2_s_kernel",
            .iq3_xxs => "paged_attention_prefill_iq3_xxs_kernel",
            .q2_k => "paged_attention_prefill_q2_k_kernel",
            .q3_k => "paged_attention_prefill_q3_k_kernel",
            .q4_k => "paged_attention_prefill_q4_k_kernel",
            .q5_k => "paged_attention_prefill_q5_k_kernel",
            .q6_k => "paged_attention_prefill_q6_k_kernel",
            else => null,
        };
        const kname = main_kernel orelse extra_kernel orelse return error.KvQuantUnsupported;
        const module = if (main_kernel != null) self.module else (self.module_extra orelse return error.KernelNotFound);

        try cudaz.ensureCurrent();

        var nq_c: c_int = @intCast(n_queries);
        _ = &nq_c;
        var sp_c: c_int = @intCast(start_pos);
        _ = &sp_c;
        var nqh_c: c_int = @intCast(config.num_q_heads);
        _ = &nqh_c;
        var nkv_c: c_int = @intCast(config.num_kv_heads);
        _ = &nkv_c;
        var hd_c: c_int = @intCast(config.head_dim);
        _ = &hd_c;
        var bs_c: c_int = @intCast(config.block_size);
        _ = &bs_c;
        var q16v = q16;
        var out16v = out16;
        var d_cache_v = try self.cacheBase(block_alloc);
        var causal_c: c_int = @intFromBool(causal);
        _ = &causal_c;

        // Lane A EXPERIMENTO DEFINITIVO: bt buffer FRESCO por llamada
        // (H2D sync inmediato). Si esto arregla el fallo, la causa es el
        // contenido de d_bts[layer] en tiempo de ejecución.
        var d_bt_used: cudaz.CUdeviceptr = undefined;
        if (bt_override) |bo| {
            d_bt_used = bo;
        } else {
            d_bt_used = try cudaz.cuMemAlloc(bt_host.len * @sizeOf(c_int));
            defer cudaz.cuMemFree(d_bt_used);
            try cudaz.cuMemcpyHtoD(d_bt_used, @intFromPtr(bt_host.ptr), bt_host.len * @sizeOf(c_int));
        }
        if (debug.dbg.dump_kv and extra_kernel != null) {
            var btrb: [2]c_int = undefined;
            cudaz.cuMemcpyDtoH(@intFromPtr(&btrb), d_bt_used, bt_host.len * @sizeOf(c_int)) catch {};
            debug.dbg.printLevel(.info, "[pre-launch bt] ptr={x} contenido=[{d},{d}] esperado=[{d},{d}]\n", .{ d_bt_used, btrb[0], btrb[1], bt_host[0], if (bt_host.len > 1) bt_host[1] else -1 });
        }

        const func = try cudaz.cuModuleGetFunction(module, kname);
        // Lane A VERIFICACIÓN FINAL: readback del bt usado JUSTO antes del kp
        if (debug.dbg.dump_kv and extra_kernel != null) {
            var btrb: [2]c_int = undefined;
            cudaz.cuStreamSynchronize(self.stream) catch {};
            cudaz.cuMemcpyDtoH(@intFromPtr(&btrb), d_bt_used, bt_host.len * @sizeOf(c_int)) catch {};
            debug.dbg.printLevel(.info, "[pre-launch bt] ptr={x} contenido=[{d},{d}] esperado=[{d},{d}]\n", .{ d_bt_used, btrb[0], btrb[1], bt_host[0], if (bt_host.len > 1) bt_host[1] else -1 });
        }
        var kp = [_]?*anyopaque{
            &out16v, &q16v, &d_cache_v, &d_bt_used,
            &nq_c,   &sp_c, &nqh_c,     &nkv_c,
            &hd_c,   &bs_c, &causal_c,
        };
        // Configurable block size for prefill (ZIG_AI_PREFILL_BLOCK_SIZE, default 32)
        const prefill_block_size: c_int = blk: {
            if (std.c.getenv("ZIG_AI_PREFILL_BLOCK_SIZE")) |env| {
                const env_slice = std.mem.span(env);
                const parsed = std.fmt.parseInt(c_int, env_slice, 10) catch 32;
                if (parsed >= 32 and parsed <= 1024 and @mod(parsed, 32) == 0) {
                    break :blk parsed;
                }
            }
            break :blk 32;
        };
        const shared_bytes: c_uint = @intCast(2 * config.head_dim * @sizeOf(f32));
        debug.dbg.printLevel(.detail, "[prefill] grid=({d},{d},1) block={d} smem={d}B\n", .{ n_queries, config.num_q_heads, prefill_block_size, shared_bytes });
        try cudaz.cuLaunchKernel(func, @intCast(n_queries), @intCast(config.num_q_heads), 1, @intCast(prefill_block_size), 1, 1, shared_bytes, self.stream, @ptrCast(&kp), null);
        if (debug.dbg.dump_kv and extra_kernel != null) {
            try cudaz.cuStreamSynchronize(self.stream);
            const chk = try self.allocator.alloc(f16, 4);
            defer self.allocator.free(chk);
            try cudaz.cuMemcpyDtoH(@intFromPtr(chk.ptr), out16v, 4 * @sizeOf(f16));
            debug.dbg.printLevel(.info, "[post-launch out] primeros: {d:.4} {d:.4} {d:.4} {d:.4}\n", .{ @as(f32, @floatCast(chk[0])), @as(f32, @floatCast(chk[1])), @as(f32, @floatCast(chk[2])), @as(f32, @floatCast(chk[3])) });
        }
    }

    /// Prefill batch: un solo lanzamiento del kernel `paged_attention_prefill_f16_kernel`
    /// (bloque por (token, q_head)) con máscara causal, en vez de iterar decode
    /// por posición. Equivale a `PagedAttention.prefill`.
    pub fn prefill(
        self: *Self,
        queries: []const f32,
        outs: []f32,
        block_table: *const BlockTable,
        block_alloc: *BlockAllocator,
        seq_len: usize,
    ) !void {
        const config = self.config;
        const num_q_heads = config.num_q_heads;
        const num_kv_heads = config.num_kv_heads;
        const head_dim = config.head_dim;
        const block_size = config.block_size;
        const q_stride = num_q_heads * head_dim;

        std.debug.assert(queries.len == seq_len * q_stride);
        std.debug.assert(outs.len == seq_len * q_stride);
        if (seq_len == 0) return;

        try cudaz.ensureCurrent();

        const total = seq_len * q_stride;
        const q_f16 = try self.allocator.alloc(f16, total);
        defer self.allocator.free(q_f16);
        for (queries, 0..) |v, i| q_f16[i] = @floatCast(v);

        const max_num_blocks = block_table.numBlocks();
        const bt_host = try self.allocator.alloc(c_int, max_num_blocks);
        defer self.allocator.free(bt_host);
        for (0..max_num_blocks) |i| {
            bt_host[i] = if (block_table.getPhysical(i)) |phys| @intCast(phys) else -1;
        }

        try self.stageBlocks(block_alloc, @constCast(block_table));

        var d_outs = try cudaz.cuMemAlloc(total * @sizeOf(f16));
        defer cudaz.cuMemFree(d_outs);
        var d_queries = try cudaz.cuMemAlloc(total * @sizeOf(f16));
        defer cudaz.cuMemFree(d_queries);
        var d_bt = try cudaz.cuMemAlloc(max_num_blocks * @sizeOf(c_int));
        defer cudaz.cuMemFree(d_bt);

        try cudaz.cuMemcpyHtoD(d_queries, @intFromPtr(q_f16.ptr), total * @sizeOf(f16));
        try cudaz.cuMemcpyHtoD(d_bt, @intFromPtr(bt_host.ptr), max_num_blocks * @sizeOf(c_int));

        const func = cudaz.cuModuleGetFunction(self.module, "paged_attention_prefill_f16_kernel") catch return error.KernelNotFound;

        var n_queries_c: c_int = @intCast(seq_len);
        var start_pos_c: c_int = 0;
        var num_q_c: c_int = @intCast(num_q_heads);
        var num_kv_c: c_int = @intCast(num_kv_heads);
        var head_dim_c: c_int = @intCast(head_dim);
        var block_size_c: c_int = @intCast(block_size);
        // 2ae7984 añadió `causal` al kernel; este wrapper legacy no lo
        // pasaba (regresión lane-a: launch de 10 args sobre firma de 11 →
        // CUDA_ERROR_INVALID_VALUE). Causal=1 (semántica original).
        var causal_c: c_int = 1;

        var d_cache_v = try self.cacheBase(block_alloc);

        var kp = [_]?*anyopaque{
            &d_outs,      &d_queries,    &d_cache_v, &d_bt,
            &n_queries_c, &start_pos_c,  &num_q_c,   &num_kv_c,
            &head_dim_c,  &block_size_c, &causal_c,
        };
        // Configurable block size for prefill (ZIG_AI_PREFILL_BLOCK_SIZE, default 32)
        const prefill_block_size: c_int = blk: {
            if (std.c.getenv("ZIG_AI_PREFILL_BLOCK_SIZE")) |env| {
                const env_slice = std.mem.span(env);
                const parsed = std.fmt.parseInt(c_int, env_slice, 10) catch 32;
                if (parsed >= 32 and parsed <= 1024 and @mod(parsed, 32) == 0) {
                    break :blk parsed;
                }
            }
            break :blk 32;
        };
        const shared_bytes: c_uint = @intCast(2 * head_dim * @sizeOf(f32));
        try cudaz.cuLaunchKernel(func, @intCast(seq_len), @intCast(num_q_heads), 1, @intCast(prefill_block_size), 1, 1, shared_bytes, self.stream, @ptrCast(&kp), null);
        try cudaz.cuStreamSynchronize(self.stream);

        const out_f16 = try self.allocator.alloc(f16, total);
        defer self.allocator.free(out_f16);
        try cudaz.cuMemcpyDtoH(@intFromPtr(out_f16.ptr), d_outs, total * @sizeOf(f16));

        for (out_f16, 0..) |v, i| outs[i] = @floatCast(v);
    }

    /// Copia bloques físicos (COW / fork) vía `block_copy_f16_kernel`.
    pub fn blockCopy(
        self: *Self,
        block_alloc: *BlockAllocator,
        copy_map: []const [2]c_int,
    ) !void {
        try cudaz.ensureCurrent();
        if (copy_map.len == 0) return;

        // Subir bloques fuente que no estén residentes.
        for (copy_map) |m| {
            const src: usize = @intCast(m[1]);
            try self.stageBlock(block_alloc, src);
        }

        const map_host = try self.allocator.alloc([2]c_int, copy_map.len);
        defer self.allocator.free(map_host);
        @memcpy(map_host, copy_map);
        var d_map = try cudaz.cuMemAlloc(map_host.len * @sizeOf([2]c_int));
        defer cudaz.cuMemFree(d_map);
        try cudaz.cuMemcpyHtoD(d_map, @intFromPtr(map_host.ptr), map_host.len * @sizeOf([2]c_int));

        var num_copies_c: c_int = @intCast(copy_map.len);
        var block_bytes_c: c_int = @intCast(block_alloc.block_bytes);
        const d_cache = try self.cacheBase(block_alloc);

        const func = cudaz.cuModuleGetFunction(self.module, "block_copy_f16_kernel") catch return error.KernelNotFound;
        var kp = [_]?*anyopaque{ &d_cache, &d_cache, &d_map, &num_copies_c, &block_bytes_c };
        const blocks: c_uint = @intCast((copy_map.len + 127) / 128);
        try cudaz.cuLaunchKernel(func, blocks, 1, 1, 128, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        try cudaz.cuStreamSynchronize(self.stream);

        // Bajar solo los bloques destino modificados.
        for (copy_map) |m| {
            const dst: usize = @intCast(m[0]);
            try self.evictBlock(block_alloc, dst);
        }
        try cudaz.cuCtxSynchronize();
    }

    /// Baja del dispositivo los bloques fríos de la prefix cache (según su
    /// hit rate) usando `PrefixCache.evictGpuCold`, liberando memoria GPU para
    /// bloques calientes. Devuelve cuántos bloques se bajaron.
    pub fn evictColdBlocksFromCache(
        self: *Self,
        block_alloc: *BlockAllocator,
        prefix_cache: *@import("prefix_cache.zig").PrefixCache,
        max_age: u64,
        min_hit_rate: f64,
    ) !usize {
        try cudaz.ensureCurrent();
        const cold = prefix_cache.evictGpuCold(max_age, min_hit_rate);
        defer self.allocator.free(cold);
        try self.evictBlocks(block_alloc, cold);
        return cold.len;
    }
};

pub const CpuReference = struct {
    pub const decode = @import("attention.zig").PagedAttention.decode;
    pub const prefill = @import("attention.zig").PagedAttention.prefill;
};
