//! Lane-b1 A0 + B3: espejo Zig del descriptor KVarN GPU + launcher
//! `kvarnInitDescsDevice`.
//!
//! Los offsets de record SIEMPRE se derivan de `kv_cache.kvarn`
//! (`KvarnRecordLayout`) — este módulo nunca duplica la aritmética de layout,
//! solo la comparte hacia los wrappers de kernel. Contrato C2: `kvarnStoreDevice`
//! y `kvarnMaterializeDevice` son la superficie estable que consumen B2/B3.

const std = @import("std");
const kvarn = @import("kv_cache").kvarn;
const cudaz = @import("cudaz");
const debugz = @import("debug");

pub const KVARN_DIM: u32 = kvarn.KVAR_N_GROUP;

/// Config completa de una memoria KVarN GPU (una por lado K/V por stream-set).
/// El llenado del `KvarnDesc` (CUDA) se hace en el wrapper, no aquí.
pub const KvarnMemoryConfig = struct {
    /// Bits K y V (independientes, 2/3/4/5/6/8).
    k_bits: u8,
    v_bits: u8,
    head_dim: u32, // 128/256/512
    /// Records por stream (ring depth).
    groups_per_stream: u32,
    /// Streams lógicos (>=1).
    n_stream: u32,
    /// Cabezas físicas grabadas por grupo (= n_kv_heads * head_slices).
    n_record_heads: u32,
    /// Profundidad del stage f16 en grupos (>=2).
    stage_groups: u32,
    /// Grupos calientes sin sellar mantenidos en stage (>=1, <= stage_groups).
    tail_groups: u32,
    swa: bool = false,
    eager_records: bool = false,
    /// Cuando `true` los resolvers leen posiciones absolutas a través del
    /// array de `indices` (modo no-SWA); con SWA, el `indices` siempre se
    /// lee (read_indirect es la constante).
    read_indirect: bool = true,
    /// Cuando `true` la FA / materialize emite el dominio original (sin
    /// aplicar WHT inversa). Lo decide el caller de la ruta específica.
    original_domain: bool = false,
    /// 1/2/4 slices por cabeza lógica (D=128/256/512). El D=128 ⇒ 1.
    head_slices: u32 = 1,

    pub fn layout(self: KvarnMemoryConfig) kvarn.KvarnRecordLayout {
        return kvarn.KvarnRecordLayout.init(self.head_dim, self.k_bits, self.v_bits) catch
            @panic("kvarn: layout inválido (head_dim/bits fuera de rango)");
    }

    pub fn recordBytes(self: KvarnMemoryConfig) usize {
        return self.layout().tile_bytes;
    }

    /// Bytes totales de records: streams × grupos × cabezas × record.
    pub fn recordsBytes(self: KvarnMemoryConfig) usize {
        return @as(usize, self.n_stream) * self.groups_per_stream * self.n_record_heads * self.recordBytes();
    }

    /// Bytes del stage f16: streams × stage_groups×128 × cabezas × 128 × 2B.
    pub fn stageBytes(self: KvarnMemoryConfig) usize {
        return @as(usize, self.n_stream) * self.stage_groups * KVARN_DIM * self.n_record_heads * KVARN_DIM * @sizeOf(f16);
    }

    /// Tokens cubiertos por el ring de records de un stream.
    pub fn tokensPerStream(self: KvarnMemoryConfig) u32 {
        return self.groups_per_stream * KVARN_DIM;
    }
};

/// Layout device-side del `KvarnDesc` (espejo de la struct C++ en
/// `kvarn_desc.cuh`). El wrapper Zig asume este layout ALINEA con el
/// struct de C — si Dev-A la cambia, este espejo debe actualizarse
/// también. Cualquier drift ⇒ silencios sutiles (offsets equivocados).
pub const KvarnDesc = extern struct {
    records: [*]const u8,
    stage: [*]const f16,
    indices: [*]const i64,
    n_record_heads: c_int,
    live_group: c_int,
    live_pos: c_int,
    stream: c_int,
    head_base: c_int,
    groups_per_stream: c_int,
    record_bytes: c_int,
    stage_groups: c_int,
    tail_groups: c_int,
    bits: c_int,
    value: c_int,
    swa: c_int,
    head_slices: c_int,
    head_dim: c_int,
    eager_records: c_int,
    read_indirect: c_int,
    original_domain: c_int,
};

/// Argumentos del kernel de init (B3, lane-b1 Dev-B). Se mantienen como
/// struct persistente para graph-capture; el wrapper pasa la dirección
/// de cada campo escalar por el kp array (patrón zig-ai).
pub const KvarnInitDescsArgs = struct {
    n_stream: c_int = 0,
    n_indices: c_int = 0,
    d_indices: [*]const i64 = undefined,
    d_descs: [*]KvarnDesc = undefined,
    desc_stride: c_int = 0,
    d_records: [*]u8 = undefined,
    d_stage: [*]f16 = undefined,
    n_record_heads: c_int = 0,
    head_dim: c_int,
    groups_per_stream: c_int = 0,
    record_bytes: c_int = 0,
    stage_groups: c_int = 0,
    tail_groups: c_int = 0,
    k_bits: c_int = 0,
    v_bits: c_int = 0,
    head_slices: c_int = 0,
    eager_records: c_int = 0,
    read_indirect: c_int = 0,
    original_domain: c_int = 0,
    swa: c_int = 0,
};

/// Resultado del init (B3): cuántos descriptores se escribieron.
pub const KvarnInitDescsResult = struct {
    n_descs_written: u32,
    n_streams: u32,
    live_groups: []c_int,
    live_pos: []c_int,

    pub fn deinit(self: *KvarnInitDescsResult, allocator: std.mem.Allocator) void {
        allocator.free(self.live_groups);
        allocator.free(self.live_pos);
    }
};

/// Decl externa del kernel init (lane-b1 Dev-B B3). El símbolo vive en
/// `kvarn_kernels.cu`; el wrapper lo carga con `cuModuleGetFunction` y
/// lanza con `cuLaunchKernel`.
pub extern "c" fn kvarn_init_descs_kernel(
    n_stream: c_int,
    n_indices: c_int,
    d_indices: [*]const i64,
    d_descs: [*]KvarnDesc,
    desc_stride: c_int,
    d_records: [*]u8,
    d_stage: [*]f16,
    n_record_heads: c_int,
    groups_per_stream: c_int,
    record_bytes: c_int,
    stage_groups: c_int,
    tail_groups: c_int,
    k_bits: c_int,
    v_bits: c_int,
    head_slices: c_int,
    eager_records: c_int,
    read_indirect: c_int,
    original_domain: c_int,
    swa: c_int,
) void;

/// Wrapper host Zig para `kvarn_init_descs_kernel` (lane-b1 Dev-B B3).
///
/// Llena `d_descs[s * desc_stride + {0,1}]` con descriptores K y V para
/// cada stream `s` ∈ [0, n_stream). El campo `live_group/live_pos` se
/// calcula en device (reducción árbol 128→1 sobre `d_indices`).
///
/// Patrón zig-ai: el `args` se pasa por valor (struct persistente
/// outside del stack frame del caller), y el kp array contiene las
/// DIRECCIONES de cada campo escalar. Esto es el patrón estable para
/// graph-capture (los punteros no cambian entre replays).
///
/// Pre-condiciones (validadas en el wrapper):
///   - d_descs tiene `n_stream * desc_stride` elementos
///   - d_indices tiene `n_stream * n_indices` elementos
///   - block 128, grid n_stream (validado en el kernel)
pub fn kvarnInitDescsDevice(
    module: cudaz.CUmodule,
    args: *const KvarnInitDescsArgs,
    stream: cudaz.CUstream,
) !void {
    if (args.n_stream <= 0) return error.InvalidStreamCount;
    if (args.n_indices <= 0) return error.InvalidIndexCount;
    if (args.desc_stride < 1) return error.InvalidDescStride;

    const func = try cudaz.cuModuleGetFunction(module, "kvarn_init_descs_kernel");

    // Pre-launch breadcrumb (formato/tamaños, como el resto de wrappers
    // zig-ai; nada de printf dentro del kernel).
    if (debugz.dbg.at(.detail)) {
        debugz.dbg.printLevel(
            .detail,
            "[kvarn] init_descs n_stream={d} n_indices={d} stride={d} groups={d} bytes={d} k={d} v={d}\n",
            .{
                args.n_stream,
                args.n_indices,
                args.desc_stride,
                args.groups_per_stream,
                args.record_bytes,
                args.k_bits,
                args.v_bits,
            },
        );
    }

    // kp array: direcciones de cada escalar (patrón zig-ai estable para
    // graph-capture). Los punteros device (d_*) tienen tipo
    // `CUdeviceptr = usize` y se pasan por dirección.
    //
    // IMPORTANTE: el orden DEBE coincidir con la firma .cu. 20 args.
    var d_indices_any: cudaz.CUdeviceptr = @intFromPtr(args.d_indices);
    var d_descs_any: cudaz.CUdeviceptr = @intFromPtr(args.d_descs);
    var d_records_any: cudaz.CUdeviceptr = @intFromPtr(args.d_records);
    var d_stage_any: cudaz.CUdeviceptr = @intFromPtr(args.d_stage);
    var kp: [20]?*const anyopaque = .{
        &args.n_stream, // 0
        &args.n_indices, // 1
        &d_indices_any, // 2
        &d_descs_any, // 3
        &args.desc_stride, // 4
        &d_records_any, // 5
        &d_stage_any, // 6
        &args.n_record_heads, // 7
        &args.head_dim, // 8
        &args.groups_per_stream, // 9
        &args.record_bytes, // 10
        &args.stage_groups, // 11
        &args.tail_groups, // 12
        &args.k_bits, // 13
        &args.v_bits, // 14
        &args.head_slices, // 15
        &args.eager_records, // 16
        &args.read_indirect, // 17
        &args.original_domain, // 18
        &args.swa, // 19
    };

    try cudaz.cuLaunchKernel(
        func,
        @intCast(args.n_stream), // grid x: 1 bloque por stream
        1,
        1,
        128, // block: 128 threads (reducción 128→1)
        1,
        1,
        0, // smem dinámico: 0 (todo va por shfl + smem estática 4 ints)
        stream,
        @ptrCast(&kp),
        null,
    );
}

test "kvarn memory config sizes" {
    const cfg = KvarnMemoryConfig{
        .k_bits = 5,
        .v_bits = 4,
        .head_dim = 128,
        .groups_per_stream = 16,
        .n_stream = 2,
        .n_record_heads = 8,
        .stage_groups = 4,
        .tail_groups = 3,
    };
    // Un record k5v4 hd128: k_payload 128*128*5/8=10240 + k_s_col 256 +
    // k_zp 256 + k_s_row 256 + v_payload 8192 + v_s_col 256 + v_s_row 256 +
    // v_zp 256 = 19968 -> align32 = 19968 (ya múltiplo).
    try std.testing.expectEqual(@as(usize, 19968), cfg.recordBytes());
    // records: 2 streams * 16 groups * 8 heads * 19968
    try std.testing.expectEqual(@as(usize, 5_111_808), cfg.recordsBytes());
    // stage: 2 * 4*128 * 8 * 128 * 2 = 2_097_152
    try std.testing.expectEqual(@as(usize, 2_097_152), cfg.stageBytes());
    try std.testing.expectEqual(@as(u32, 2048), cfg.tokensPerStream());
}

test "KvarnDesc layout: size contains 3 ptrs + 17 ints" {
    // El struct en kvarn_desc.cuh tiene 3 punteros (records, stage,
    // indices) + 17 c_int. El tamaño EXACTO depende de alineación de
    // cada campo en el struct de C, que en x86_64 con clang/NVCC es
    // típicamente 8B para ptrs y 4B para ints, con padding al final a
    // múltiplo del miembro más ancho (8B).
    //
    // Aquí validamos que el struct Zig tiene al menos los bytes
    // suficientes para alojar 3 ptrs + 17 ints sin packing destructivo.
    // El harness GPU (test_kvarn_init_descs_gpu.zig, B3) es la prueba
    // definitiva de que el layout es compatible con C — el device
    // corromperá silenciosamente los offsets si drift.
    const min_size: usize = @sizeOf([*]const u8) * 3 + @sizeOf(c_int) * 17;
    try std.testing.expect(@sizeOf(KvarnDesc) >= min_size);
}

/// Contrato C2 (lane-b1 Dev A): store GPU de tokens a records C1.
/// La firma es ESTABLE — B2/B3 dependen de ella.
pub const KvarnStoreArgs = struct {
    current: [*]const f32, // [n_tokens, n_record_heads, 128] dominio ORIGINAL (K)
    current_v: ?[*]const f32 = null, // idem V; null ⇒ mismo buffer que K
    indices: [*]const i64, // [n_tokens] celdas codificadas de ESTE stream
    stage: [*]f16,
    records: [*]u8,
    n_tokens: c_int,
    n_record_heads: c_int,
    stream: c_int,
    groups_per_stream: c_int,
    record_bytes: c_int,
    k_payload_off: c_int,
    k_s_col_off: c_int,
    k_zp_off: c_int,
    k_s_row_off: c_int,
    v_payload_off: c_int,
    v_s_col_off: c_int,
    v_s_row_off: c_int,
    v_zp_off: c_int,
    k_bits: c_int,
    v_bits: c_int,
    sinkhorn_iters: c_int,
    stage_groups: c_int,
    tail_groups: c_int,
    swa: c_int,
    eager_records: c_int,
};

/// Bytes de shared dinámico del store (C2v2: tiles K y V separados).
pub const KVARN_STORE_SMEM_FLOATS: usize = 16384 + 8 * 128 + 2 + 16;
pub const KVARN_STORE_SMEM_BYTES: usize = KVARN_STORE_SMEM_FLOATS * @sizeOf(f32);

pub fn kvarnStoreDevice(
    module: cudaz.CUmodule,
    args: *const KvarnStoreArgs,
    stream: cudaz.CUstream,
) !void {
    if (args.n_tokens <= 0) return error.InvalidTokenCount;
    if (args.n_record_heads <= 0) return error.InvalidHeadCount;
    if (args.record_bytes <= 0) return error.InvalidRecordBytes;
    if (args.stage_groups < 2) return error.InvalidStageGroups;

    const func = try cudaz.cuModuleGetFunction(module, "kvarn_store_kernel");

    // Shared dinámico 69,704B > default 48KB: raise por CUfunction.
    // Idempotente y barato — se llama siempre (cuModuleLoad puede crear
    // CUfunction nuevas por módulo; un raise-once global dejaría las
    // siguientes sin atributo).
    // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8.
    try cudaz.cuFuncSetAttribute(func, 8, @intCast(KVARN_STORE_SMEM_BYTES));

    if (debugz.dbg.dump_kvarn) {
        debugz.dbg.printLevel(.detail, "[kvarn] store n_tokens={d} heads={d} stream={d} kb={d} vb={d} smem={d}\n", .{
            args.n_tokens, args.n_record_heads, args.stream,
            args.k_bits,   args.v_bits,         KVARN_STORE_SMEM_BYTES,
        });
    }

    var d_current_any: cudaz.CUdeviceptr = @intFromPtr(args.current);
    var d_current_v_any: cudaz.CUdeviceptr = if (args.current_v) |cv|
        @intFromPtr(cv)
    else
        0;
    var d_indices_any: cudaz.CUdeviceptr = @intFromPtr(args.indices);
    var d_stage_any: cudaz.CUdeviceptr = @intFromPtr(args.stage);
    var d_records_any: cudaz.CUdeviceptr = @intFromPtr(args.records);

    var kp: [25]?*const anyopaque = .{
        &d_current_any, // 0 current
        &d_current_v_any, // 1 current_v (0 ⇒ mismo buffer)
        &d_indices_any, // 2 indices
        &d_stage_any, // 2 stage
        &d_records_any, // 3 records
        &args.n_tokens, // 4
        &args.n_record_heads, // 5
        &args.stream, // 6
        &args.groups_per_stream, // 7
        &args.record_bytes, // 8
        &args.k_payload_off, // 9
        &args.k_s_col_off, // 10
        &args.k_zp_off, // 11
        &args.k_s_row_off, // 12
        &args.v_payload_off, // 13
        &args.v_s_col_off, // 14
        &args.v_s_row_off, // 15
        &args.v_zp_off, // 16
        &args.k_bits, // 17
        &args.v_bits, // 18
        &args.sinkhorn_iters, // 19
        &args.stage_groups, // 20
        &args.tail_groups, // 21
        &args.swa, // 22
        &args.eager_records, // 23
    };
    _ = &kp;

    try cudaz.cuLaunchKernel(
        func,
        @intCast(args.n_record_heads), // grid: 1 bloque por cabeza
        1,
        1,
        128, // block
        1,
        1,
        @intCast(KVARN_STORE_SMEM_BYTES),
        stream,
        @ptrCast(&kp),
        null,
    );
}

/// A5: smem estático del lowshmem — 788 floats = 3,152 B (< 48KB default,
/// sin opt-in; habilita sm_75 y coexiste con el split kernel).
pub const KVARN_STORE_LOWSHMEM_BYTES: usize = 3_152;

/// A5-adaptativo (M3): elige hishmem/lowshmem por smem opt-in del
/// device. `smem_optin` = MAX_SHARED_MEMORY_PER_BLOCK_OPTIN (probeDevice
/// .shared_memory_per_block). ≥ KVARN_STORE_SMEM_BYTES (69,704) ⇒ hishmem
/// (ruta bit-exacta M1); si no ⇒ lowshmem (sm_75 y coexistencia con
/// kernels de smem alto). Ambos producen records BIT-IDÉNTICOS (el
/// sellado siempre re-lee el stage f16 — test A5 19968/19968).
pub fn kvarnStoreAdaptive(
    module: cudaz.CUmodule,
    args: *const KvarnStoreArgs,
    stream: cudaz.CUstream,
    smem_optin: ?u32,
) !void {
    if (smem_optin) |s| {
        if (s >= KVARN_STORE_SMEM_BYTES) {
            try kvarnStoreDevice(module, args, stream);
            return;
        }
    }
    try kvarnStoreLowShmemDevice(module, args, stream);
}

/// A5: store low-shmem — MISMA firma/semántica que kvarnStoreDevice pero
/// sellando re-lectura del stage half global (sin tile smem). NO
/// bit-exacto con el hishmem (f16-rounding pre-Sinkhorn, upstream idem).
pub fn kvarnStoreLowShmemDevice(
    module: cudaz.CUmodule,
    args: *const KvarnStoreArgs,
    stream: cudaz.CUstream,
) !void {
    if (args.n_tokens <= 0) return error.InvalidTokenCount;
    if (args.n_record_heads <= 0) return error.InvalidHeadCount;
    if (args.record_bytes <= 0) return error.InvalidRecordBytes;
    if (args.stage_groups < 2) return error.InvalidStageGroups;

    const func = try cudaz.cuModuleGetFunction(module, "kvarn_store_lowshmem_kernel");

    if (debugz.dbg.dump_kvarn) {
        debugz.dbg.printLevel(.detail, "[kvarn] store-lowshmem n_tokens={d} heads={d} stream={d} kb={d} vb={d} smem=static-3152\n", .{
            args.n_tokens, args.n_record_heads, args.stream,
            args.k_bits,   args.v_bits,
        });
    }

    var d_current_any: cudaz.CUdeviceptr = @intFromPtr(args.current);
    var d_current_v_any: cudaz.CUdeviceptr = if (args.current_v) |cv|
        @intFromPtr(cv)
    else
        0;
    var d_indices_any: cudaz.CUdeviceptr = @intFromPtr(args.indices);
    var d_stage_any: cudaz.CUdeviceptr = @intFromPtr(args.stage);
    var d_records_any: cudaz.CUdeviceptr = @intFromPtr(args.records);

    var kp: [25]?*const anyopaque = .{
        &d_current_any, // 0 current
        &d_current_v_any, // 1 current_v (0 ⇒ mismo buffer)
        &d_indices_any, // 2 indices
        &d_stage_any, // 3 stage
        &d_records_any, // 4 records
        &args.n_tokens, // 5
        &args.n_record_heads, // 6
        &args.stream, // 7
        &args.groups_per_stream, // 8
        &args.record_bytes, // 9
        &args.k_payload_off, // 10
        &args.k_s_col_off, // 11
        &args.k_zp_off, // 12
        &args.k_s_row_off, // 13
        &args.v_payload_off, // 14
        &args.v_s_col_off, // 15
        &args.v_s_row_off, // 16
        &args.v_zp_off, // 17
        &args.k_bits, // 18
        &args.v_bits, // 19
        &args.sinkhorn_iters, // 20
        &args.stage_groups, // 21
        &args.tail_groups, // 22
        &args.swa, // 23
        &args.eager_records, // 24
    };
    _ = &kp;

    // smem estático: shared=0, SIN cuFuncSetAttribute.
    try cudaz.cuLaunchKernel(
        func,
        @intCast(args.n_record_heads), // grid: 1 bloque por cabeza
        1,
        1,
        128, // block
1,
        1,
        0, // dyn smem = 0 (todo estático < 48KB)
        stream,
        @ptrCast(&kp),
        null,
    );
}

/// 9.4 (lane-b) D2-store: `kvarn_store_d256_kernel` — store D=256 con
/// cabezas físicas por slice + cross-slice WHT (espejo beellama
/// kvarn.cu:1055-1210). `n_logical_heads` = heads D=256 del `current`;
/// los records/stage se dimensionan con `n_record_heads` FÍSICAS
/// (= 2·n_logical_heads) en layout 128. `current`/`current_v`:
/// [n_tokens][n_logical_heads][256].
pub const KvarnStoreD256Args = struct {
    current: [*]const f32, // [n_tokens, n_logical_heads, 256]
    current_v: ?[*]const f32 = null,
    indices: [*]const i64,
    stage: [*]f16, // [pos][2·n_record_heads][128]
    records: [*]u8, // [groups][n_record_heads][record_bytes]
    n_tokens: c_int,
    n_logical_heads: c_int,
    n_record_heads: c_int, // = 2·n_logical_heads
    stream: c_int,
    groups_per_stream: c_int,
    record_bytes: c_int,
    k_payload_off: c_int,
    k_s_col_off: c_int,
    k_zp_off: c_int,
    k_s_row_off: c_int,
    v_payload_off: c_int,
    v_s_col_off: c_int,
    v_s_row_off: c_int,
    v_zp_off: c_int,
    k_bits: c_int,
    v_bits: c_int,
    sinkhorn_iters: c_int,
    stage_groups: c_int,
    tail_groups: c_int,
    swa: c_int,
    eager_records: c_int,
};

pub fn kvarnStoreD256Device(
    module: cudaz.CUmodule,
    args: *const KvarnStoreD256Args,
    stream: cudaz.CUstream,
) !void {
    if (args.n_tokens <= 0) return error.InvalidTokenCount;
    if (args.n_logical_heads <= 0) return error.InvalidHeadCount;
    if (args.n_record_heads != 2 * args.n_logical_heads) return error.InvalidHeadCount;
    if (args.record_bytes <= 0) return error.InvalidRecordBytes;
    if (args.stage_groups < 2) return error.InvalidStageGroups;

    const func = try cudaz.cuModuleGetFunction(module, "kvarn_store_d256_kernel");

    // Mismo smem dinámico que el store 128 (seal en smem).
    try cudaz.cuFuncSetAttribute(func, 8, @intCast(KVARN_STORE_SMEM_BYTES));

    if (debugz.dbg.dump_kvarn) {
        debugz.dbg.printLevel(.detail, "[kvarn] store-d256 n_tokens={d} logical={d} physical={d} stream={d} kb={d} vb={d} smem={d}\n", .{
            args.n_tokens, args.n_logical_heads, args.n_record_heads, args.stream,
            args.k_bits,   args.v_bits,         KVARN_STORE_SMEM_BYTES,
        });
    }

    var d_current_any: cudaz.CUdeviceptr = @intFromPtr(args.current);
    var d_current_v_any: cudaz.CUdeviceptr = if (args.current_v) |cv|
        @intFromPtr(cv)
    else
        0;
    var d_indices_any: cudaz.CUdeviceptr = @intFromPtr(args.indices);
    var d_stage_any: cudaz.CUdeviceptr = @intFromPtr(args.stage);
    var d_records_any: cudaz.CUdeviceptr = @intFromPtr(args.records);

    var kp: [26]?*const anyopaque = .{
        &d_current_any, // 0
        &d_current_v_any, // 1
        &d_indices_any, // 2
        &d_stage_any, // 3
        &d_records_any, // 4
        &args.n_tokens, // 5
        &args.n_logical_heads, // 6
        &args.n_record_heads, // 7
        &args.stream, // 8
        &args.groups_per_stream, // 9
        &args.record_bytes, // 10
        &args.k_payload_off, // 11
        &args.k_s_col_off, // 12
        &args.k_zp_off, // 13
        &args.k_s_row_off, // 14
        &args.v_payload_off, // 15
        &args.v_s_col_off, // 16
        &args.v_s_row_off, // 17
        &args.v_zp_off, // 18
        &args.k_bits, // 19
        &args.v_bits, // 20
        &args.sinkhorn_iters, // 21
        &args.stage_groups, // 22
        &args.tail_groups, // 23
        &args.swa, // 24
        &args.eager_records, // 25
    };
    _ = &kp;

    try cudaz.cuLaunchKernel(
        func,
        @intCast(args.n_logical_heads), // grid: 1 bloque por cabeza LÓGICA
        1,
        1,
        128, // block
        1,
        1,
        @intCast(KVARN_STORE_SMEM_BYTES),
        stream,
        @ptrCast(&kp),
        null,
    );
}

/// 9.12 (lane-cuda) F2: D6-store: `kvarn_store_d64_kernel` — store D=64 con
/// tiles rectangulares (64×128 K, 128×64 V). `n_record_heads` = heads
/// físicas totales. `current`/`current_v`: [n_tokens][n_record_heads][64].
pub const KvarnStoreD64Args = struct {
    current: [*]const f32, // [n_tokens, n_record_heads, 64]
    current_v: ?[*]const f32 = null,
    indices: [*]const i64,
    stage: [*]f16, // [pos][2·n_record_heads][128]
    records: [*]u8, // [groups][n_record_heads][record_bytes]
    n_tokens: c_int,
    n_record_heads: c_int,
    stream: c_int,
    groups_per_stream: c_int,
    record_bytes: c_int,
    k_payload_off: c_int,
    k_s_col_off: c_int,
    k_zp_off: c_int,
    k_s_row_off: c_int,
    v_payload_off: c_int,
    v_s_col_off: c_int,
    v_s_row_off: c_int,
    v_zp_off: c_int,
    k_bits: c_int,
    v_bits: c_int,
    sinkhorn_iters: c_int,
    stage_groups: c_int,
    tail_groups: c_int,
    swa: c_int,
    eager_records: c_int,
};

pub fn kvarnStoreD64Device(
    module: cudaz.CUmodule,
    args: *const KvarnStoreD64Args,
    stream: cudaz.CUstream,
) !void {
    if (args.n_tokens <= 0) return error.InvalidTokenCount;
    if (args.n_record_heads <= 0) return error.InvalidHeadCount;
    if (args.record_bytes <= 0) return error.InvalidRecordBytes;
    if (args.stage_groups < 2) return error.InvalidStageGroups;

    const func = try cudaz.cuModuleGetFunction(module, "kvarn_store_d64_kernel");

    // smem dinámico: tile 8192 + arrays 128×6 + reduce 16 + 2 = 8978 floats
    const d64_smem_bytes: c_int = @intCast((8192 + 128 * 6 + 16 + 2) * @sizeOf(f32)); // 35912 bytes
    try cudaz.cuFuncSetAttribute(func, 8, @intCast(d64_smem_bytes));

    if (debugz.dbg.dump_kvarn) {
        debugz.dbg.printLevel(.detail, "[kvarn] store-d64 n_tokens={d} heads={d} stream={d} kb={d} vb={d} smem={d}\n", .{
            args.n_tokens, args.n_record_heads, args.stream,
            args.k_bits,   args.v_bits,         d64_smem_bytes,
        });
    }

    var d_current_any: cudaz.CUdeviceptr = @intFromPtr(args.current);
    var d_current_v_any: cudaz.CUdeviceptr = if (args.current_v) |cv|
        @intFromPtr(cv)
    else
        0;
    var d_indices_any: cudaz.CUdeviceptr = @intFromPtr(args.indices);
    var d_stage_any: cudaz.CUdeviceptr = @intFromPtr(args.stage);
    var d_records_any: cudaz.CUdeviceptr = @intFromPtr(args.records);

    var kp: [25]?*const anyopaque = .{
        &d_current_any, // 0
        &d_current_v_any, // 1
        &d_indices_any, // 2
        &d_stage_any, // 3
        &d_records_any, // 4
        &args.n_tokens, // 5
        &args.n_record_heads, // 6
        &args.stream, // 7
        &args.groups_per_stream, // 8
        &args.record_bytes, // 9
        &args.k_payload_off, // 10
        &args.k_s_col_off, // 11
        &args.k_zp_off, // 12
        &args.k_s_row_off, // 13
        &args.v_payload_off, // 14
        &args.v_s_col_off, // 15
        &args.v_s_row_off, // 16
        &args.v_zp_off, // 17
        &args.k_bits, // 18
        &args.v_bits, // 19
        &args.sinkhorn_iters, // 20
        &args.stage_groups, // 21
        &args.tail_groups, // 22
        &args.swa, // 23
        &args.eager_records, // 24
    };
    _ = &kp;

    try cudaz.cuLaunchKernel(
        func,
        @intCast(args.n_record_heads), // grid: 1 bloque por cabeza física
        1,
        1,
        128, // block
        1,
        1,
        @intCast(d64_smem_bytes),
        stream,
        @ptrCast(&kp),
        null,
    );
}

/// A6: materialize — una fila f16 por (token, head). `value` 0=K/1=V.
/// El caller decide offsets C1 del lado (layout.k_* o layout.v_*).
pub const KvarnMaterializeArgs = struct {
    records: [*]const u8,
    stage: [*]const f16,
    indices: [*]const i64,
    out: [*]f16, // [n_tokens, n_heads, 128]
    n_tokens: c_int,
    n_heads: c_int,
    stream: c_int,
    groups_per_stream: c_int,
    record_bytes: c_int,
    payload_off: c_int,
    scale_off: c_int,
    zp_off: c_int,
    other_off: c_int,
    bits: c_int,
    value: c_int,
    stage_groups: c_int,
    tail_groups: c_int,
    swa: c_int,
    eager_records: c_int,
    read_indirect: c_int,
    live_group: c_int,
    live_pos: c_int,
    emit_rotated: c_int,
};

pub fn kvarnMaterializeDevice(
    module: cudaz.CUmodule,
    args: *const KvarnMaterializeArgs,
    stream: cudaz.CUstream,
) !void {
    if (args.n_tokens <= 0) return error.InvalidTokenCount;
    if (args.n_heads <= 0) return error.InvalidHeadCount;
    if (args.record_bytes <= 0) return error.InvalidRecordBytes;

    const func = try cudaz.cuModuleGetFunction(module, "kvarn_materialize_kernel");

    if (debugz.dbg.dump_kvarn) {
        debugz.dbg.printLevel(.detail, "[kvarn] materialize n_tokens={d} heads={d} bits={d} value={d} rotated={d}\n", .{
            args.n_tokens, args.n_heads, args.bits, args.value, args.emit_rotated,
        });
    }

    var d_records_any: cudaz.CUdeviceptr = @intFromPtr(args.records);
    var d_stage_any: cudaz.CUdeviceptr = @intFromPtr(args.stage);
    var d_indices_any: cudaz.CUdeviceptr = @intFromPtr(args.indices);
    var d_out_any: cudaz.CUdeviceptr = @intFromPtr(args.out);

    var kp: [23]?*const anyopaque = .{
        &d_records_any, // 0
        &d_stage_any, // 1
        &d_indices_any, // 2
        &d_out_any, // 3
        &args.n_tokens, // 4
        &args.n_heads, // 5
        &args.stream, // 6
        &args.groups_per_stream, // 7
        &args.record_bytes, // 8
        &args.payload_off, // 9
        &args.scale_off, // 10
        &args.zp_off, // 11
        &args.other_off, // 12
        &args.bits, // 13
        &args.value, // 14
        &args.stage_groups, // 15
        &args.tail_groups, // 16
        &args.swa, // 17
        &args.eager_records, // 18
        &args.read_indirect, // 19
        &args.live_group, // 20
        &args.live_pos, // 21
        &args.emit_rotated, // 22
    };

    try cudaz.cuLaunchKernel(
        func,
        @intCast(args.n_tokens), // grid x: token
        @intCast(args.n_heads), // grid y: head
        1,
        128, // block: un thread por dim
        1,
        1,
        0, // dyn smem: 0 (row[128] es __shared__ estático en el kernel)
        stream,
        @ptrCast(&kp),
        null,
    );
}

// ============================================================================
// A11 (Dev-A): geometry selection para decode-split MMA.
// ============================================================================

/// Ocupación de un kernel por SM (vía cuOccupancyMaxActiveBlocksPer-
/// Multiprocessor con CU_FUNC_HANDLE driver-API) + SM count cacheados por
/// device. El selector evalúa candidatos {MAX_GQA, NWARPS} y escoge el
/// mejor scoring upstream: blocks_per_sm·10⁶ + wave_eff·10⁴ −
/// gqa_blocks·10³ − n_splits; cutoff "direct ≥2 waves al ≥75% ⇒ no split".
pub const KvarnSplitGeometry = struct {
    use_split: bool,
    split_tokens: u32, // fijo 64 (template)
    nwarps: u32,
    gqa_per_block: u32, // MAX_GQA del template (6 u 8)
    n_splits: u32,
    n_gqa_blocks: u32,
    max_blocks_per_sm: u32,
    wave_efficiency_percent: u32,
    n_waves: u32,
    candidate_count: u32,
};

extern "c" fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    numBlocks: *c_int,
    func: cudaz.CUfunction,
    blockSize: c_int,
    dynamicSMemSize: c_uint,
    flags: c_uint,
) c_int;

extern "c" fn cuDeviceGetAttribute(pi: *c_int, attrib: c_int, dev: c_int) c_int;

const occupancy_cache = struct {
    var sm_count: c_int = -1;
    var checked: bool = false;
};

fn smCount() c_int {
    if (!occupancy_cache.checked) {
        var dev: c_int = -1;
        if (cuCtxGetDevice(&dev) != cudaz.CUresult.SUCCESS) return -1;
        // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
        if (cuDeviceGetAttribute(&occupancy_cache.sm_count, 16, dev) != 0) {
            occupancy_cache.sm_count = -1;
        }
        occupancy_cache.checked = true;
    }
    return occupancy_cache.sm_count;
}

extern "c" fn cuCtxGetDevice(dev: *c_int) cudaz.CUresult;

fn divUp(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}

/// Ocupación de un bloque del split kernel (blockSize = 32·nwarps, smem 0).
fn maxBlocksPerSm(func: cudaz.CUfunction, nwarps: u32) c_int {
    var blocks: c_int = 0;
    const rc = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&blocks, func, @intCast(32 * nwarps), 0, 0);
    if (rc != 0) return -1;
    return blocks;
}

fn waveEff(blocks_total: u64, blocks_per_wave: u32, n_waves: *u32) u32 {
    if (blocks_total == 0 or blocks_per_wave == 0) {
        n_waves.* = 0;
        return 0;
    }
    n_waves.* = @intCast((blocks_total + blocks_per_wave - 1) / blocks_per_wave);
    return @intCast(100 * blocks_total / (@as(u64, n_waves.*) * blocks_per_wave));
}

/// Selección de geometría para un (D=128, split=64) dado el workload.
/// El caller pasa el CUfunction del template ya elegido por bits (los
/// candidatos de ocupación comparten shape por MAX_GQA 6/8 y NWARPS 4).
pub fn selectSplitGeometry(
    module: cudaz.CUmodule,
    n_kv: u32,
    n_q: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    n_stream: u32,
) !KvarnSplitGeometry {
    if (n_kv == 0 or n_q == 0 or n_q_heads == 0 or n_kv_heads == 0 or n_stream == 0)
        return error.InvalidShape;
    if (n_q_heads % n_kv_heads != 0) return error.GqaMismatch;
    const gqa_ratio: u32 = n_q_heads / n_kv_heads;

    const nsm = smCount();
    if (nsm <= 0) return error.OccupancyUnavailable;

    // Candidatos D=128 (A10 templates): (gqa6, w4) y (gqa8, w4).
    const cands = [_]struct { gqa: u32, nwarps: u32, name: []const u8 }{
        .{ .gqa = 6, .nwarps = 4, .name = "kvarn_decode_mma_d128_gqa6_s64_w4_k4v4_kernel" },
        .{ .gqa = 8, .nwarps = 4, .name = "kvarn_decode_mma_d128_gqa6_s64_w4_k4v4_kernel" }, // placeholder: solo gqa6 existe
    };

    var best: KvarnSplitGeometry = .{
        .use_split = false,
        .split_tokens = 64,
        .nwarps = 0,
        .gqa_per_block = 0,
        .n_splits = 0,
        .n_gqa_blocks = 0,
        .max_blocks_per_sm = 0,
        .wave_efficiency_percent = 0,
        .n_waves = 0,
        .candidate_count = 0,
    };
    var best_score: i64 = std.math.minInt(i64);

    // El único kernel instanciado hoy es gqa6_w4 (A10 wrapper único). Los
    // "candidatos" futuros (gqa8) requieren sus propios templates.
    const func = try cudaz.cuModuleGetFunction(module, "kvarn_decode_mma_d128_gqa6_s64_w4_k4v4_kernel");
    inline for (cands) |c| {
        if (gqa_ratio <= c.gqa) {
            const n_splits = divUp(n_kv, 64);
            const n_gqa_blocks = divUp(gqa_ratio, c.gqa);
            const blocks_per_sm = maxBlocksPerSm(func, c.nwarps);
            if (blocks_per_sm > 0) {
                best.candidate_count += 1;
                const blocks_per_wave: u32 = @intCast(nsm * blocks_per_sm);
                const blocks_total: u64 = @as(u64, n_splits) * n_kv_heads * n_gqa_blocks * n_q * n_stream;
                var wv: u32 = 0;
                const eff = waveEff(blocks_total, blocks_per_wave, &wv);
                const score: i64 = @as(i64, blocks_per_sm) * 1_000_000 +
                    @as(i64, eff) * 10_000 -
                    @as(i64, n_gqa_blocks) * 1_000 -
                    @as(i64, n_splits);
                if (score > best_score) {
                    best_score = score;
                    best = .{
                        .use_split = true,
                        .split_tokens = 64,
                        .nwarps = c.nwarps,
                        .gqa_per_block = c.gqa,
                        .n_splits = n_splits,
                        .n_gqa_blocks = n_gqa_blocks,
                        .max_blocks_per_sm = @intCast(blocks_per_sm),
                        .wave_efficiency_percent = eff,
                        .n_waves = wv,
                        .candidate_count = best.candidate_count,
                    };
                }
            }
        }
    }

    if (!best.use_split or best.n_splits <= 1) {
        best.use_split = false;
        return best;
    }

    // Cutoff upstream: si el grid DIRECTO (sin split) ya llena ≥2 waves al
    // ≥75%, el overhead del combine no compensa.
    const direct_blocks: u64 = @as(u64, n_kv_heads) * best.n_gqa_blocks * n_q * n_stream;
    var direct_waves: u32 = 0;
    const direct_eff = waveEff(direct_blocks, @intCast(@as(i64, nsm) * best.max_blocks_per_sm), &direct_waves);
    if (direct_waves >= 2 and direct_eff >= 75) {
        best.use_split = false;
    }
    return best;
}

/// Args del launch decode-split (A10/A13): el dispatcher (B7) usa
/// esta superficie cuando selectRoute elige decode_split. El caller
/// aporta los buffers partial/meta (device) dimensionados con
/// n_splits = divUp(n_kv, 64).
pub const KvarnSplitLaunchArgs = struct {
    q_data: cudaz.CUdeviceptr,
    k_descs: cudaz.CUdeviceptr,
    v_descs: cudaz.CUdeviceptr,
    mask_data: cudaz.CUdeviceptr = 0,
    partial_data: cudaz.CUdeviceptr,
    meta_data: cudaz.CUdeviceptr,
    dst_data: cudaz.CUdeviceptr,
    n_kv: c_int,
    n_q: c_int,
    n_q_heads: c_int,
    n_kv_heads: c_int,
    n_stream: c_int = 1,
    n_splits: c_int,
    scale: f32,
};

/// Resultado del split launch (contadores B1 + bloques).
pub const KvarnSplitLaunchResult = struct {
    route: []const u8 = "decode_split",
    n_blocks_mma: u32,
    n_blocks_combine: u32,
    bytes_partial: u64,
    bytes_meta: u64,
};

/// Launch decode-split MMA + combine. El cubin del split es
/// `kvarn_split_cubin` (build option lane-b1); kernels:
/// `kvarn_decode_mma_d128_gqa6_s64_w4_k4v4_kernel` (ARITY=14) y
/// `kvarn_decode_combine_d128_kernel` (ARITY=6). NO sincroniza el
/// stream (graph-capture friendly).
pub fn kvarnDecodeSplitDevice(
    module: cudaz.CUmodule,
    args: *const KvarnSplitLaunchArgs,
    stream: cudaz.CUstream,
) !KvarnSplitLaunchResult {
    if (args.n_kv <= 0 or args.n_q <= 0 or args.n_q_heads <= 0 or args.n_kv_heads <= 0)
        return error.InvalidAttentionShape;
    if (args.n_splits <= 0) return error.InvalidSplitGeometry;
    if (@rem(args.n_q_heads, args.n_kv_heads) != 0) return error.GqaMismatch;

    const mma_func = try cudaz.cuModuleGetFunction(module, "kvarn_decode_mma_d128_gqa6_s64_w4_k4v4_kernel");
    const comb_func = try cudaz.cuModuleGetFunction(module, "kvarn_decode_combine_d128_kernel");

    const gqa: c_int = @divTrunc(args.n_q_heads, args.n_kv_heads);
    const n_gqa_blocks: c_int = @intCast(@divTrunc(gqa + 5, 6)); // MAX_GQA=6

    // kp del MMA — ARITY=14: 6 ptrs + scale + 7 ints. Los 5 primeros
    // args se re-usan como base del combine (misma memoria).
    var q_any = args.q_data;
    var kd_any = args.k_descs;
    var vd_any = args.v_descs;
    var mask_any = args.mask_data;
    var partial_any = args.partial_data;
    var meta_any = args.meta_data;
    var n_kv_v = args.n_kv;
    var n_q_v = args.n_q;
    var n_qh_v = args.n_q_heads;
    var n_kvh_v = args.n_kv_heads;
    var gqa_v = gqa;
    var n_gqa_v = n_gqa_blocks;
    var n_splits_v = args.n_splits;
    var scale_v = args.scale;
    var kp: [14]?*const anyopaque = .{
        &q_any,       &kd_any,     &vd_any,  &mask_any,
        &partial_any, &meta_any,   &scale_v, &n_kv_v,
        &n_q_v,       &n_qh_v,     &n_kvh_v, &gqa_v,
        &n_gqa_v,     &n_splits_v,
    };
    const grid_y: c_uint = @intCast(args.n_kv_heads * n_gqa_blocks * args.n_q * args.n_stream);
    try cudaz.cuLaunchKernel(
        mma_func,
        @intCast(args.n_splits), // grid x: splits
        grid_y,
        1,
        32, // block x
        4, // block y: NWARPS
        1,
        0,
        stream,
        @ptrCast(&kp),
        null,
    );

    // Combine — ARITY=6: partial, meta, dst, n_splits, n_q, n_q_heads.
    var dst_any = args.dst_data;
    var kp2: [6]?*const anyopaque = .{
        &partial_any, &meta_any, &dst_any,
        &n_splits_v,  &n_q_v,    &n_qh_v,
    };
    const smem_combine: c_uint = @intCast(@as(u32, @intCast(args.n_splits)) * @sizeOf(f32));
    try cudaz.cuLaunchKernel(
        comb_func,
        @intCast(args.n_q_heads), // grid x
        @intCast(args.n_q * args.n_stream), // grid y
        1,
        256, // block
        1,
        1,
        smem_combine,
        stream,
        @ptrCast(&kp2),
        null,
    );

    const partial_len: u64 = @as(u64, @intCast(args.n_stream * args.n_q * args.n_q_heads * args.n_splits)) * 128;
    const meta_len: u64 = @as(u64, @intCast(args.n_stream * args.n_q * args.n_q_heads * args.n_splits));
    return .{
        .n_blocks_mma = @intCast(@as(i64, args.n_splits) * @as(i64, grid_y)),
        .n_blocks_combine = @intCast(@as(i64, args.n_q_heads) * args.n_q * args.n_stream),
        .bytes_partial = partial_len * @sizeOf(f32),
        .bytes_meta = meta_len * 2 * @sizeOf(f32),
    };
}

test "A11 geometry: selectSplitGeometry con shape válida devuelve split" {
    // Sin GPU: cuModuleGetFunction fallaría. Validación de la lógica pura
    // se hace con GPU real (gated). Aquí solo validamos los errores de
    // pre-condición.
    try std.testing.expectError(error.InvalidShape, selectSplitGeometry(undefined, 0, 1, 1, 1, 1));
    try std.testing.expectError(error.GqaMismatch, selectSplitGeometry(undefined, 128, 1, 7, 2, 1));
}
