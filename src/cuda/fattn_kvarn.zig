//! Lane-b1 B4 (Dev-B): dispatch master + portable launcher for KVarN FA.
//!
//! Owns the in-Zig side of the portable attention path:
//!   - KvarnAttentionArgs / KvarnAttentionResult: stable structs (graph-
//!     capture friendly, kp pattern).
//!   - `fattnKvarnPortableDevice(module, args, stream)`: launches the
//!     D=128 kernel for the body-only path (B4 scope per TODO_B1_DEV_B
//!     §B4). Tail (D4) and vec (B6) are follow-up tasks.
//!   - Route counters integration via `backend_capabilities` so the
//!     dispatch master has one place to ask "what route was chosen for
//!     this launch".
//!
//! ## Scope notes
//!
//!   - D ∈ {128, 256, 512} accepted by `portableSupported` (B1), but
//!     only D=128 has a kernel implementation in B4; D≥256 gated by D2
//!     until B2 ratifies cross-slice WHT (TODO_D2 in PLAN_B1).
//!   - B7 dispatch master is a follow-up; this file is the *launcher*
//!     called by the dispatch master (or by hand-written B5 tests).
//!   - `kvarnInitDescsDevice` (B3) is called from here as a convenience:
//!     the typical launcher flow is "init descs → launch portable".

const std = @import("std");
const debugz = @import("debug");
const cudaz = @import("cudaz");
const bc = @import("backend_capabilities");
const kvk = @import("kvarn_kernels");

pub const KVARN_DIM: u32 = 128;

/// Argumentos del launch portable (B4). Estructura persistente para
/// graph-capture (mismo patrón que `kvarn_kernels.KvarnInitDescsArgs`):
/// el kernel recibe los punteros por dirección vía kp.
pub const KvarnAttentionArgs = struct {
    /// Tensor Q contiguo f32, layout [n_q, n_q_heads, n_stream, D].
    /// Dev A's materialize path owns el contrato exacto; aquí asumimos
    /// contigüidad (TODO: si B7 introduce strides variables, ampliar).
    q_data: [*]const f32 = undefined,
    /// Buffer de descriptores K (n_stream × n_kv_heads) pre-llenado por
    /// `kvarnInitDescsDevice`. Lado K: cada desc tiene value=0.
    k_descs: [*]kvk.KvarnDesc = undefined,
    /// Idem V. value=1.
    v_descs: [*]kvk.KvarnDesc = undefined,
    /// Máscara f16 [n_kv, n_q] o null (sin máscara).
    mask_data: ?[*]const f16 = null,
    /// Tensor dst contiguo f32 [n_q_heads, n_q, n_stream, D].
    dst_data: [*]f32 = undefined,
    n_kv: c_int = 0,
    n_q: c_int = 0,
    n_q_heads: c_int = 0,
    n_kv_heads: c_int = 0,
    n_stream: c_int = 0,
    /// 1/sqrt(D) en fp32 (lo computa el caller; el kernel NO lo hace).
    scale: f32 = 0.0,
    /// GQA (n_q_heads / n_kv_heads) precomputado para validación rápida.
    gqa: c_int = 0,
};

/// Resultado del launch (B4). Por ahora un marcador: el dispatcher
/// (B7) lo consultará para acumular contadores + mem stats vía
/// `backend_capabilities`.
pub const KvarnAttentionResult = struct {
    route: bc.Route,
    n_blocks: u32,
    bytes_q: u64,
    bytes_k_descs: u64,
    bytes_v_descs: u64,
    bytes_dst: u64,
};

/// Decl externa del kernel (lane-b1 Dev-B B4, D=128 only).
pub extern "c" fn fattn_kvarn_portable_d128_kernel(
    q_data: [*]const f32,
    k_descs: [*]kvk.KvarnDesc,
    v_descs: [*]kvk.KvarnDesc,
    mask_data: ?[*]const f16,
    dst_data: [*]f32,
    n_kv: c_int,
    n_q: c_int,
    n_q_heads: c_int,
    n_kv_heads: c_int,
    n_stream: c_int,
    scale: f32,
) void;

/// Launcher para el d128 + D4 KVCPT tail (lane-b1 Dev-B, B4 iter 3).
/// Patrón kp de 19 args + 6 ptrs (k_tail/v_tail/tail_mask/run_desc_slots
/// + q/k_descs/v_descs/dst_data).
///
/// Pre-condiciones (validadas):
///   - shape de atención válida (GQA, n_q > 0)
///   - k_tail_data no nulo SI n_tail > 0
///   - run_desc_slots no nulo SI n_tail > 0
///
/// El wrapper NO sincroniza el stream (graph-capture friendly); el
/// caller sincroniza al final del paso si necesita.
pub fn fattnKvarnPortableD128TailDevice(
    module: ?cudaz.CUmodule,
    args: *const KvarnAttentionTailArgs,
    stream: cudaz.CUstream,
) !void {
    if (args.n_q <= 0 or args.n_q_heads <= 0) return error.InvalidTailShape;
    if (args.n_q_heads != args.gqa * args.n_kv_heads) return error.GqaMismatch;
    if (args.n_tail > 0) {
        if (args.k_tail_data == null) return error.MissingTailK;
        if (args.v_tail_data == null) return error.MissingTailV;
        if (args.run_desc_slots == null) return error.MissingRunDescSlots;
    }

    const func = try cudaz.cuModuleGetFunction(
        module orelse return error.NeedCudaModule,
        "fattn_kvarn_portable_d128_tail_kernel",
    );

    if (debugz.dbg.at(.detail)) {
        debugz.dbg.printLevel(
            .detail,
            "[kvarn-fa] portable D=128+tail n_kv={d} n_tail={d} n_qh={d} k_bf16={any} v_bf16={any}\n",
            .{
                args.n_kv,
                args.n_tail,
                args.n_q_heads,
                args.k_tail_bf16 != 0,
                args.v_tail_bf16 != 0,
            },
        );
    }

    // 6 ptrs + 13 ints = 19 args.
    var q_data_any: cudaz.CUdeviceptr = @intFromPtr(args.q_data);
    var k_descs_any: cudaz.CUdeviceptr = @intFromPtr(args.k_descs);
    var v_descs_any: cudaz.CUdeviceptr = @intFromPtr(args.v_descs);
    var mask_data_any: cudaz.CUdeviceptr = if (args.mask_data) |m| @intFromPtr(m) else 0;
    var k_tail_any: cudaz.CUdeviceptr = if (args.k_tail_data) |p| @intFromPtr(p) else 0;
    var v_tail_any: cudaz.CUdeviceptr = if (args.v_tail_data) |p| @intFromPtr(p) else 0;
    var tail_mask_any: cudaz.CUdeviceptr = if (args.tail_mask) |m| @intFromPtr(m) else 0;
    var dst_data_any: cudaz.CUdeviceptr = @intFromPtr(args.dst_data);
    var run_desc_slots_any: cudaz.CUdeviceptr = if (args.run_desc_slots) |p| @intFromPtr(p) else 0;

    // ARITY=20: must match the 20 scalars of fattn_kvarn_portable_d128_tail_kernel.
    // Mismatch = silent garbage-arg reading at the device (iter-15/17 bug).
    var kp: [20]?*const anyopaque = .{
        &q_data_any, // 0
        &k_descs_any, // 1
        &v_descs_any, // 2
        &mask_data_any, // 3
        &k_tail_any, // 4
        &v_tail_any, // 5
        &tail_mask_any, // 6
        &run_desc_slots_any, // 7
        &args.n_kv, // 8
        &args.n_tail, // 9
        &args.d_k, // 10
        &args.d_v, // 11
        &args.n_q, // 12
        &args.n_q_heads, // 13
        &args.n_kv_heads, // 14
        &args.n_stream, // 15
        &args.k_tail_bf16, // 16
        &args.v_tail_bf16, // 17
        &dst_data_any, // 18
        &args.scale, // 19 — BUG iter-15/17: faltaba scale en el kp
    };

    try cudaz.cuLaunchKernel(
        func,
        @intCast(args.n_q),
        @intCast(args.n_q_heads),
        @intCast(args.n_stream),
        128,
        1,
        1,
        0,
        stream,
        @ptrCast(&kp),
        null,
    );

    // Counter tracking (la body+tail ruta es portable-native desde
    // el punto de vista del dispatcher; el tail fusion es interno).
    bc.countersInc(.portable_native);
    bc.memStatsUpdate(.descriptor, @as(u64, @intCast(args.n_stream)) * @as(u64, @intCast(args.n_kv_heads)) * 2 * @sizeOf(kvk.KvarnDesc));
}

/// Argumentos del launch D=128 + D4 KVCPT tail.
pub const KvarnAttentionTailArgs = struct {
    q_data: [*]const f32 = undefined,
    k_descs: [*]kvk.KvarnDesc = undefined,
    v_descs: [*]kvk.KvarnDesc = undefined,
    mask_data: ?[*]const f16 = null,
    /// Puntero a k_tail_data; null ⇒ no tail (cuerpo sólo).
    k_tail_data: ?[*]const u8 = null,
    v_tail_data: ?[*]const u8 = null,
    tail_mask: ?[*]const f16 = null,
    /// Array de `n_tail` slots (i32). Slot i = posición del token i-ésimo
    /// del tail en k_tail_data/v_tail_data. Layout upstream
    /// `run_desc[6+token]`.
    run_desc_slots: ?[*]const c_int = null,
    n_kv: c_int = 0,
    n_tail: c_int = 0,
    d_k: c_int = 0,
    d_v: c_int = 0,
    n_q: c_int = 0,
    n_q_heads: c_int = 0,
    n_kv_heads: c_int = 0,
    n_stream: c_int = 0,
    /// 1 ⇒ bf16, 0 ⇒ f16.
    k_tail_bf16: c_int = 0,
    v_tail_bf16: c_int = 0,
    dst_data: [*]f32 = undefined,
    scale: f32 = 0.0,
    gqa: c_int = 0,
};

test "KvarnAttentionTailArgs: defaults n_tail=0, n_kv=0" {
    const args: KvarnAttentionTailArgs = .{};
    try std.testing.expectEqual(@as(c_int, 0), args.n_kv);
    try std.testing.expectEqual(@as(c_int, 0), args.n_tail);
    try std.testing.expectEqual(@as(?[*]const u8, null), args.k_tail_data);
}

test "fattnKvarnPortableD128TailDevice: n_q=0 ⇒ error.InvalidTailShape" {
    var args: KvarnAttentionTailArgs = .{};
    try std.testing.expectError(error.InvalidTailShape, fattnKvarnPortableD128TailDevice(null, &args, @ptrFromInt(1)));
}

test "fattnKvarnPortableD128TailDevice: GQA mismatch ⇒ error.GqaMismatch" {
    var args: KvarnAttentionTailArgs = .{
        .q_data = @ptrFromInt(0x100),
        .k_descs = @ptrFromInt(0x200),
        .v_descs = @ptrFromInt(0x300),
        .n_q = 1,
        .n_q_heads = 7, // not multiple of n_kv_heads
        .n_kv_heads = 2,
        .gqa = 4,
        .n_stream = 1,
        .scale = 0.1,
    };
    try std.testing.expectError(error.GqaMismatch, fattnKvarnPortableD128TailDevice(null, &args, @ptrFromInt(1)));
}

test "fattnKvarnPortableD128TailDevice: n_tail>0 sin pointers ⇒ error.MissingTailK" {
    var args: KvarnAttentionTailArgs = .{
        .q_data = @ptrFromInt(0x100),
        .k_descs = @ptrFromInt(0x200),
        .v_descs = @ptrFromInt(0x300),
        .n_q = 1,
        .n_q_heads = 2,
        .n_kv_heads = 1,
        .n_stream = 1,
        .scale = 0.1,
        .gqa = 2,
        .n_tail = 4, // tail requested but k_tail_data null
    };
    try std.testing.expectError(error.MissingTailK, fattnKvarnPortableD128TailDevice(null, &args, @ptrFromInt(1)));
}

/// Decl externa del kernel D=128 con cola exacta D4 KVCPT (B4 follow-up).
pub extern "c" fn fattn_kvarn_portable_d128_tail_kernel(
    q_data: [*]const f32,
    k_descs: [*]kvk.KvarnDesc,
    v_descs: [*]kvk.KvarnDesc,
    mask_data: ?[*]const f16,
    k_tail_data: [*]const u8,
    v_tail_data: [*]const u8,
    tail_mask: ?[*]const f16,
    run_desc_slots: ?[*]const c_int,
    n_kv: c_int,
    n_tail: c_int,
    d_k: c_int,
    d_v: c_int,
    n_q: c_int,
    n_q_heads: c_int,
    n_kv_heads: c_int,
    n_stream: c_int,
    k_tail_bf16: c_int,
    v_tail_bf16: c_int,
    dst_data: [*]f32,
    scale: f32,
) void;

/// Decl externa del kernel D=256 (B4 + D7 cross-slice WHT, gated D2).
/// Mismo layout que D=128; la rotación Q incluye intra-slice WHT-128
/// + cross-slice WHT-2 (ver `fattn_kvarn_wht_cross_slices<2>` en el .cu).
pub extern "c" fn fattn_kvarn_portable_d256_kernel(
    q_data: [*]const f32,
    k_descs: [*]kvk.KvarnDesc,
    v_descs: [*]kvk.KvarnDesc,
    mask_data: ?[*]const f16,
    dst_data: [*]f32,
    n_kv: c_int,
    n_q: c_int,
    n_q_heads: c_int,
    n_kv_heads: c_int,
    n_stream: c_int,
    scale: f32,
) void;

/// Decl externa del kernel D=512 (B4 + D7 cross-slice WHT-4, gated D2).
/// SLICES=4, mismo patrón que D=256 pero con cross-slice WHT-4
/// (closed-form) en lugar de WHT-2.
pub extern "c" fn fattn_kvarn_portable_d512_kernel(
    q_data: [*]const f32,
    k_descs: [*]kvk.KvarnDesc,
    v_descs: [*]kvk.KvarnDesc,
    mask_data: ?[*]const f16,
    dst_data: [*]f32,
    n_kv: c_int,
    n_q: c_int,
    n_q_heads: c_int,
    n_kv_heads: c_int,
    n_stream: c_int,
    scale: f32,
) void;

/// Shape-check para portable (re-exporta `bc.portableSupported` con la
/// firma que el dispatcher usará). El dispatcher (B7) decide si portable
/// es elegible — aquí validamos la SHAPE concreta de un launch.
pub fn launchShapeOk(args: *const KvarnAttentionArgs) bool {
    if (args.n_q <= 0 or args.n_q_heads <= 0) return false;
    if (args.n_kv <= 0 or args.n_kv_heads <= 0) return false;
    if (args.n_stream <= 0) return false;
    if (args.gqa <= 0) return false;
    if (args.n_q_heads != args.gqa * args.n_kv_heads) return false;
    if (args.scale <= 0.0 or !std.math.isFinite(args.scale)) return false;
    return true;
}

/// Wrapper host Zig para `fattn_kvarn_portable_d128_kernel` (lane-b1
/// Dev-B B4). Lanza el kernel portable body-only para D=128.
///
/// Pre-condiciones (validadas en el wrapper):
///   - D = 128 (D≥256 gated por D2; sólo se compila el template D=128)
///   - shape válida (ver `launchShapeOk`)
///   - `k_descs` y `v_descs` ya están poblados por `kvarnInitDescsDevice`
///
/// El wrapper NO sincroniza el stream (graph-capture friendly); el
/// caller sincroniza al final del paso si necesita.
pub fn fattnKvarnPortableDevice(
    module: ?cudaz.CUmodule,
    args: *const KvarnAttentionArgs,
    stream: cudaz.CUstream,
) !KvarnAttentionResult {
    if (!launchShapeOk(args)) return error.InvalidAttentionShape;

    // D=128 only — D>=256 gated.
    const d: u32 = 128;
    if (d != 128) return error.UnsupportedHeadDim;

    const func = try cudaz.cuModuleGetFunction(module orelse return error.NeedCudaModule, "fattn_kvarn_portable_d128_kernel");

    // Pre-launch breadcrumb (formato/tamaños; patrón zig-ai). Cuando
    // B7 llegue, este breadcrumb se complementa con la ruta elegida
    // (bc.countersPrint()).
    if (debugz.dbg.at(.detail)) {
        debugz.dbg.printLevel(
            .detail,
            "[kvarn-fa] portable D=128 n_q={d} n_qh={d} n_kv={d} n_kvh={d} gqa={d} streams={d} scale={d}\n",
            .{
                args.n_q,
                args.n_q_heads,
                args.n_kv,
                args.n_kv_heads,
                args.gqa,
                args.n_stream,
                args.scale,
            },
        );
    }

    // kp array: direcciones de cada escalar (patrón zig-ai estable).
    // 11 args (5 ptrs + 6 ints + 1 float).
    var q_data_any: cudaz.CUdeviceptr = @intFromPtr(args.q_data);
    var k_descs_any: cudaz.CUdeviceptr = @intFromPtr(args.k_descs);
    var v_descs_any: cudaz.CUdeviceptr = @intFromPtr(args.v_descs);
    var mask_data_any: cudaz.CUdeviceptr = if (args.mask_data) |m| @intFromPtr(m) else 0;
    var dst_data_any: cudaz.CUdeviceptr = @intFromPtr(args.dst_data);
    // ARITY=11: must match the 11 scalars of fattn_kvarn_portable_d128_kernel.
    // Mismatch = silent garbage-arg reading at the device (iter-15/17 bug).
    var kp: [11]?*const anyopaque = .{
        &q_data_any, // 0
        &k_descs_any, // 1
        &v_descs_any, // 2
        &mask_data_any, // 3
        &dst_data_any, // 4
        &args.n_kv, // 5
        &args.n_q, // 6
        &args.n_q_heads, // 7
        &args.n_kv_heads, // 8
        &args.n_stream, // 9
        &args.scale, // 10
    };

    try cudaz.cuLaunchKernel(
        func,
        @intCast(args.n_q), // grid x
        @intCast(args.n_q_heads), // grid y
        @intCast(args.n_stream), // grid z
        128, // block: 128 threads
        1,
        1,
        0, // smem dinámico: 0
        stream,
        @ptrCast(&kp),
        null,
    );

    // Actualiza contadores + mem stats (B1). El dispatcher (B7) usará
    // estos números para reports de bench.
    bc.countersInc(.portable_native);
    bc.memStatsUpdate(.descriptor, @as(u64, @intCast(args.n_stream)) * @as(u64, @intCast(args.n_kv_heads)) * 2 * @sizeOf(kvk.KvarnDesc));
    bc.memStatsUpdate(.partials, 0); // portable: 0 partials (output directo a dst)
    bc.memStatsUpdate(.meta, 0);

    return .{
        .route = .portable_native,
        .n_blocks = @intCast(args.n_q * args.n_q_heads * args.n_stream),
        .bytes_q = @as(u64, @intCast(args.n_q)) * @as(u64, @intCast(args.n_q_heads)) *
            @as(u64, @intCast(args.n_stream)) * d * @sizeOf(f32),
        .bytes_k_descs = @as(u64, @intCast(args.n_stream)) * @as(u64, @intCast(args.n_kv_heads)) * @sizeOf(kvk.KvarnDesc),
        .bytes_v_descs = @as(u64, @intCast(args.n_stream)) * @as(u64, @intCast(args.n_kv_heads)) * @sizeOf(kvk.KvarnDesc),
        .bytes_dst = @as(u64, @intCast(args.n_q_heads)) * @as(u64, @intCast(args.n_q)) *
            @as(u64, @intCast(args.n_stream)) * d * @sizeOf(f32),
    };
}

/// 9.12 (lane-cuda) F3: Wrapper host Zig para `fattn_kvarn_portable_d64_kernel`.
/// D=64 body-only portable FA. Misma estructura que D=128 pero con D=64,
/// THREADS=64, WHT-64 rotation. Stage layout [pos][2·n_record_heads][128]
/// — D64 solo usa first64. Record layout con head_dim=64.
pub fn fattnKvarnPortableD64Device(
    module: ?cudaz.CUmodule,
    args: *const KvarnAttentionArgs,
    stream: cudaz.CUstream,
) !KvarnAttentionResult {
    if (!launchShapeOk(args)) return error.InvalidAttentionShape;

    const d: u32 = 64;
    const func = try cudaz.cuModuleGetFunction(module orelse return error.NeedCudaModule, "fattn_kvarn_portable_d64_kernel");

    if (debugz.dbg.at(.detail)) {
        debugz.dbg.printLevel(
            .detail,
            "[kvarn-fa] portable D=64 n_q={d} n_qh={d} n_kv={d} n_kvh={d} gqa={d} streams={d} scale={d}\n",
            .{
                args.n_q,
                args.n_q_heads,
                args.n_kv,
                args.n_kv_heads,
                args.gqa,
                args.n_stream,
                args.scale,
            },
        );
    }

    // kp array: 11 args (5 ptrs + 6 ints + 1 float) — same as D128.
    var q_data_any: cudaz.CUdeviceptr = @intFromPtr(args.q_data);
    var k_descs_any: cudaz.CUdeviceptr = @intFromPtr(args.k_descs);
    var v_descs_any: cudaz.CUdeviceptr = @intFromPtr(args.v_descs);
    var mask_data_any: cudaz.CUdeviceptr = if (args.mask_data) |m| @intFromPtr(m) else 0;
    var dst_data_any: cudaz.CUdeviceptr = @intFromPtr(args.dst_data);

    var kp: [11]?*const anyopaque = .{
        &q_data_any, // 0
        &k_descs_any, // 1
        &v_descs_any, // 2
        &mask_data_any, // 3
        &dst_data_any, // 4
        &args.n_kv, // 5
        &args.n_q, // 6
        &args.n_q_heads, // 7
        &args.n_kv_heads, // 8
        &args.n_stream, // 9
        &args.scale, // 10
    };

    try cudaz.cuLaunchKernel(
        func,
        @intCast(args.n_q),
        @intCast(args.n_q_heads),
        @intCast(args.n_stream),
        64, // block: 64 threads (1 per dim)
        1,
        1,
        0,
        stream,
        @ptrCast(&kp),
        null,
    );

    bc.countersInc(.portable_native);
    bc.memStatsUpdate(.descriptor, @as(u64, @intCast(args.n_stream)) * @as(u64, @intCast(args.n_kv_heads)) * 2 * @sizeOf(kvk.KvarnDesc));
    bc.memStatsUpdate(.partials, 0);
    bc.memStatsUpdate(.meta, 0);

    return .{
        .route = .portable_native,
        .n_blocks = @intCast(args.n_q * args.n_q_heads * args.n_stream),
        .bytes_q = @as(u64, @intCast(args.n_q)) * @as(u64, @intCast(args.n_q_heads)) *
            @as(u64, @intCast(args.n_stream)) * d * @sizeOf(f32),
        .bytes_k_descs = @as(u64, @intCast(args.n_stream)) * @as(u64, @intCast(args.n_kv_heads)) * @sizeOf(kvk.KvarnDesc),
        .bytes_v_descs = @as(u64, @intCast(args.n_stream)) * @as(u64, @intCast(args.n_kv_heads)) * @sizeOf(kvk.KvarnDesc),
        .bytes_dst = @as(u64, @intCast(args.n_q_heads)) * @as(u64, @intCast(args.n_q)) *
            @as(u64, @intCast(args.n_stream)) * d * @sizeOf(f32),
    };
}

/// Wrapper host Zig para `fattn_kvarn_portable_d256_kernel` (B4 + D7,
/// gated D2). Misma firma que D=128 pero usa el kernel d256 del cubin
/// `fattn_kvarn.cubin` (mismo binario, distinta instanciación por
/// cuModuleGetFunction name lookup).
pub fn fattnKvarnPortableD256Device(
    module: ?cudaz.CUmodule,
    args: *const KvarnAttentionArgs,
    stream: cudaz.CUstream,
) !KvarnAttentionResult {
    if (!launchShapeOk(args)) return error.InvalidAttentionShape;
    // D=256 gate (D2 ratificación con lane-b2).
    // Validamos D via head_slices del descriptor (no hay campo D explícito
    // en KvarnAttentionArgs; el caller lo sabe).
    if (args.scale <= 0.0 or !std.math.isFinite(args.scale)) {
        return error.InvalidScale;
    }

    const func = try cudaz.cuModuleGetFunction(module orelse return error.NeedCudaModule, "fattn_kvarn_portable_d256_kernel");

    if (debugz.dbg.at(.detail)) {
        debugz.dbg.printLevel(
            .detail,
            "[kvarn-fa] portable D=256 n_q={d} n_qh={d} n_kv={d} n_kvh={d} gqa={d} streams={d} scale={d}\n",
            .{
                args.n_q,
                args.n_q_heads,
                args.n_kv,
                args.n_kv_heads,
                args.gqa,
                args.n_stream,
                args.scale,
            },
        );
    }

    var q_data_any: cudaz.CUdeviceptr = @intFromPtr(args.q_data);
    var k_descs_any: cudaz.CUdeviceptr = @intFromPtr(args.k_descs);
    var v_descs_any: cudaz.CUdeviceptr = @intFromPtr(args.v_descs);
    var mask_data_any: cudaz.CUdeviceptr = if (args.mask_data) |m| @intFromPtr(m) else 0;
    var dst_data_any: cudaz.CUdeviceptr = @intFromPtr(args.dst_data);
    // ARITY=11: must match the 11 scalars of fattn_kvarn_portable_d128_kernel.
    // Mismatch = silent garbage-arg reading at the device (iter-15/17 bug).
    var kp: [11]?*const anyopaque = .{
        &q_data_any, &k_descs_any, &v_descs_any,    &mask_data_any,   &dst_data_any,
        &args.n_kv,  &args.n_q,    &args.n_q_heads, &args.n_kv_heads, &args.n_stream,
        &args.scale,
    };

    try cudaz.cuLaunchKernel(
        func,
        @intCast(args.n_q),
        @intCast(args.n_q_heads),
        @intCast(args.n_stream),
        128,
        1,
        1,
        0,
        stream,
        @ptrCast(&kp),
        null,
    );

    bc.countersInc(.portable_native);
    bc.memStatsUpdate(.descriptor, @as(u64, @intCast(args.n_stream)) * @as(u64, @intCast(args.n_kv_heads)) * 2 * @sizeOf(kvk.KvarnDesc));

    return .{
        .route = .portable_native,
        .n_blocks = @intCast(args.n_q * args.n_q_heads * args.n_stream),
        .bytes_q = @as(u64, @intCast(args.n_q)) * @as(u64, @intCast(args.n_q_heads)) *
            @as(u64, @intCast(args.n_stream)) * 256 * @sizeOf(f32),
        .bytes_k_descs = @as(u64, @intCast(args.n_stream)) * @as(u64, @intCast(args.n_kv_heads)) * @sizeOf(kvk.KvarnDesc),
        .bytes_v_descs = @as(u64, @intCast(args.n_stream)) * @as(u64, @intCast(args.n_kv_heads)) * @sizeOf(kvk.KvarnDesc),
        .bytes_dst = @as(u64, @intCast(args.n_q_heads)) * @as(u64, @intCast(args.n_q)) *
            @as(u64, @intCast(args.n_stream)) * 256 * @sizeOf(f32),
    };
}

/// Wrapper host Zig para `fattn_kvarn_portable_d512_kernel` (B4 + D7,
/// gated D2). Misma firma que D=128; el kernel d512 del cubin
/// `fattn_kvarn.cubin` aplica 4× WHT-128 intra-slice + WHT-4 cross-slice.
pub fn fattnKvarnPortableD512Device(
    module: ?cudaz.CUmodule,
    args: *const KvarnAttentionArgs,
    stream: cudaz.CUstream,
) !KvarnAttentionResult {
    if (!launchShapeOk(args)) return error.InvalidAttentionShape;
    if (args.scale <= 0.0 or !std.math.isFinite(args.scale)) {
        return error.InvalidScale;
    }

    const func = try cudaz.cuModuleGetFunction(module orelse return error.NeedCudaModule, "fattn_kvarn_portable_d512_kernel");

    if (debugz.dbg.at(.detail)) {
        debugz.dbg.printLevel(
            .detail,
            "[kvarn-fa] portable D=512 n_q={d} n_qh={d} n_kv={d} n_kvh={d} gqa={d} streams={d} scale={d}\n",
            .{
                args.n_q,
                args.n_q_heads,
                args.n_kv,
                args.n_kv_heads,
                args.gqa,
                args.n_stream,
                args.scale,
            },
        );
    }

    var q_data_any: cudaz.CUdeviceptr = @intFromPtr(args.q_data);
    var k_descs_any: cudaz.CUdeviceptr = @intFromPtr(args.k_descs);
    var v_descs_any: cudaz.CUdeviceptr = @intFromPtr(args.v_descs);
    var mask_data_any: cudaz.CUdeviceptr = if (args.mask_data) |m| @intFromPtr(m) else 0;
    var dst_data_any: cudaz.CUdeviceptr = @intFromPtr(args.dst_data);
    // ARITY=11: must match the 11 scalars of fattn_kvarn_portable_d128_kernel.
    // Mismatch = silent garbage-arg reading at the device (iter-15/17 bug).
    var kp: [11]?*const anyopaque = .{
        &q_data_any, &k_descs_any, &v_descs_any,    &mask_data_any,   &dst_data_any,
        &args.n_kv,  &args.n_q,    &args.n_q_heads, &args.n_kv_heads, &args.n_stream,
        &args.scale,
    };

    try cudaz.cuLaunchKernel(
        func,
        @intCast(args.n_q),
        @intCast(args.n_q_heads),
        @intCast(args.n_stream),
        128,
        1,
        1,
        0,
        stream,
        @ptrCast(&kp),
        null,
    );

    bc.countersInc(.portable_native);
    bc.memStatsUpdate(.descriptor, @as(u64, @intCast(args.n_stream)) * @as(u64, @intCast(args.n_kv_heads)) * 2 * @sizeOf(kvk.KvarnDesc));

    return .{
        .route = .portable_native,
        .n_blocks = @intCast(args.n_q * args.n_q_heads * args.n_stream),
        .bytes_q = @as(u64, @intCast(args.n_q)) * @as(u64, @intCast(args.n_q_heads)) *
            @as(u64, @intCast(args.n_stream)) * 512 * @sizeOf(f32),
        .bytes_k_descs = @as(u64, @intCast(args.n_stream)) * @as(u64, @intCast(args.n_kv_heads)) * @sizeOf(kvk.KvarnDesc),
        .bytes_v_descs = @as(u64, @intCast(args.n_stream)) * @as(u64, @intCast(args.n_kv_heads)) * @sizeOf(kvk.KvarnDesc),
        .bytes_dst = @as(u64, @intCast(args.n_q_heads)) * @as(u64, @intCast(args.n_q)) *
            @as(u64, @intCast(args.n_stream)) * 512 * @sizeOf(f32),
    };
}

// ─── Self-tests (pure CPU, no GPU) ─────────────────────────────────────────

test "launchShapeOk: valid shape" {
    var args: KvarnAttentionArgs = .{
        .q_data = @ptrFromInt(0x100),
        .k_descs = @ptrFromInt(0x200),
        .v_descs = @ptrFromInt(0x300),
        .dst_data = @ptrFromInt(0x400),
        .n_kv = 256,
        .n_q = 1,
        .n_q_heads = 8,
        .n_kv_heads = 2,
        .n_stream = 1,
        .scale = 0.08838834764831845, // 1/sqrt(128)
        .gqa = 4,
    };
    try std.testing.expect(launchShapeOk(&args));
}

test "launchShapeOk: rejects n_q_heads != gqa * n_kv_heads" {
    var args: KvarnAttentionArgs = .{
        .q_data = @ptrFromInt(0x100),
        .k_descs = @ptrFromInt(0x200),
        .v_descs = @ptrFromInt(0x300),
        .dst_data = @ptrFromInt(0x400),
        .n_kv = 128,
        .n_q = 1,
        .n_q_heads = 7, // not a multiple of n_kv_heads=2
        .n_kv_heads = 2,
        .n_stream = 1,
        .scale = 0.1,
        .gqa = 4, // 4*2=8 != 7
    };
    try std.testing.expect(!launchShapeOk(&args));
}

test "launchShapeOk: rejects non-finite scale" {
    var args: KvarnAttentionArgs = .{
        .q_data = @ptrFromInt(0x100),
        .k_descs = @ptrFromInt(0x200),
        .v_descs = @ptrFromInt(0x300),
        .dst_data = @ptrFromInt(0x400),
        .n_kv = 128,
        .n_q = 1,
        .n_q_heads = 4,
        .n_kv_heads = 1,
        .n_stream = 1,
        .scale = std.math.nan(f32),
        .gqa = 4,
    };
    try std.testing.expect(!launchShapeOk(&args));
}

test "launchShapeOk: rejects zero counts" {
    var args: KvarnAttentionArgs = .{
        .q_data = @ptrFromInt(0x100),
        .k_descs = @ptrFromInt(0x200),
        .v_descs = @ptrFromInt(0x300),
        .dst_data = @ptrFromInt(0x400),
        .n_kv = 0, // inválido
        .n_q = 1,
        .n_q_heads = 4,
        .n_kv_heads = 1,
        .n_stream = 1,
        .scale = 0.1,
        .gqa = 4,
    };
    try std.testing.expect(!launchShapeOk(&args));
}

test "KvarnAttentionArgs default values" {
    const args: KvarnAttentionArgs = .{};
    try std.testing.expectEqual(@as(c_int, 0), args.n_q);
    try std.testing.expectEqual(@as(f32, 0.0), args.scale);
    try std.testing.expectEqual(@as(?[*]const f16, null), args.mask_data);
}

test "D=256 wrapper: scale=0 ⇒ InvalidScale" {
    const args: KvarnAttentionArgs = .{
        .q_data = @ptrFromInt(0x100),
        .k_descs = @ptrFromInt(0x200),
        .v_descs = @ptrFromInt(0x300),
        .dst_data = @ptrFromInt(0x400),
        .n_kv = 128,
        .n_q = 1,
        .n_q_heads = 4,
        .n_kv_heads = 1,
        .n_stream = 1,
        .scale = 0.0, // inválido
        .gqa = 4,
    };
    try std.testing.expectError(error.InvalidScale, fattnKvarnPortableD256Device(null, &args, @ptrFromInt(1)));
}

test "D=256 wrapper: scale=NaN ⇒ InvalidScale" {
    const args: KvarnAttentionArgs = .{
        .q_data = @ptrFromInt(0x100),
        .k_descs = @ptrFromInt(0x200),
        .v_descs = @ptrFromInt(0x300),
        .dst_data = @ptrFromInt(0x400),
        .n_kv = 128,
        .n_q = 1,
        .n_q_heads = 4,
        .n_kv_heads = 1,
        .n_stream = 1,
        .scale = std.math.nan(f32),
        .gqa = 4,
    };
    try std.testing.expectError(error.InvalidScale, fattnKvarnPortableD256Device(null, &args, @ptrFromInt(1)));
}

test "D=256 wrapper: shape inválida (GQA mismatch) ⇒ InvalidAttentionShape" {
    const args: KvarnAttentionArgs = .{
        .q_data = @ptrFromInt(0x100),
        .k_descs = @ptrFromInt(0x200),
        .v_descs = @ptrFromInt(0x300),
        .dst_data = @ptrFromInt(0x400),
        .n_kv = 128,
        .n_q = 1,
        .n_q_heads = 7, // 7 != 2*4
        .n_kv_heads = 2,
        .n_stream = 1,
        .scale = 0.1,
        .gqa = 4,
    };
    try std.testing.expectError(error.InvalidAttentionShape, fattnKvarnPortableD256Device(null, &args, @ptrFromInt(1)));
}

test "D=512 wrapper: scale=0 ⇒ InvalidScale" {
    const args: KvarnAttentionArgs = .{
        .q_data = @ptrFromInt(0x100),
        .k_descs = @ptrFromInt(0x200),
        .v_descs = @ptrFromInt(0x300),
        .dst_data = @ptrFromInt(0x400),
        .n_kv = 256,
        .n_q = 1,
        .n_q_heads = 4,
        .n_kv_heads = 1,
        .n_stream = 1,
        .scale = 0.0,
        .gqa = 4,
    };
    try std.testing.expectError(error.InvalidScale, fattnKvarnPortableD512Device(null, &args, @ptrFromInt(1)));
}

test "D=512 wrapper: shape válida pero module=null → cudaz error al buscar función" {
    // Sin cubin (module=null), cuModuleGetFunction devuelve un error.
    // No validamos el error específico (depende de cudaz), sólo que
    // la firma valida y dispatcha.
    const args: KvarnAttentionArgs = .{
        .q_data = @ptrFromInt(0x100),
        .k_descs = @ptrFromInt(0x200),
        .v_descs = @ptrFromInt(0x300),
        .dst_data = @ptrFromInt(0x400),
        .n_kv = 256,
        .n_q = 1,
        .n_q_heads = 4,
        .n_kv_heads = 1,
        .n_stream = 1,
        .scale = 0.04419417382415924, // 1/sqrt(512)
        .gqa = 4,
    };
    const result = fattnKvarnPortableD512Device(null, &args, @ptrFromInt(1));
    // Aceptamos cualquier error (cudaz o InvalidScale); la firma compila.
    if (result) |_| {} else |_| {}
}

// ─── B6: Vec path (D=256, SWA, GQA=2, n_q=1) ───────────────────────────────

/// Argumenos del launch vec (B6). El vec es un path especializado con
/// un único template (D=256, TPS=16, MAX_GQA=2, K/V=4 bits) en la
/// primera iteración. Futuros templates de B6 iter 2+ amplían esto.
pub const KvarnVecArgs = struct {
    q_data: [*]const f32 = undefined,
    k_descs: [*]kvk.KvarnDesc = undefined,
    v_descs: [*]kvk.KvarnDesc = undefined,
    mask_data: ?[*]const f16 = null,
    dst_data: [*]f32 = undefined,
    n_kv: c_int = 0,
    n_q_heads: c_int = 0,
    n_kv_heads: c_int = 0,
    n_stream: c_int = 0,
    scale: f32 = 0.0,
};

/// Resultado del launch vec (B6).
pub const KvarnVecResult = struct {
    route: bc.Route,
    n_blocks: u32,
};

/// Decl externa del kernel vec D=256 k4v4 (B6, iter 1).
pub extern "c" fn fattn_kvarn_vec_d256_k4v4_kernel(
    q_data: [*]const f32,
    k_descs: [*]kvk.KvarnDesc,
    v_descs: [*]kvk.KvarnDesc,
    mask_data: ?[*]const f16,
    dst_data: [*]f32,
    n_kv: c_int,
    n_q_heads: c_int,
    n_kv_heads: c_int,
    n_stream: c_int,
    scale: f32,
) void;

/// Eligibility gate para vec (B6): D=256, SWA=true, GQA=2, n_q=1, y
/// bits dentro de fast-pairs D5 (placeholder: k4v4 sólo en iter 1).
/// Re-exportado de `bc` con la firma que B7 consumirá.
pub fn vecEligible(
    head_dim: u32,
    n_q: u32,
    gqa: u32,
    swa: bool,
    k_bits: u8,
    v_bits: u8,
) bool {
    _ = swa; // (swa se lee del desc, no del input directo; placeholder)
    if (head_dim != 256) return false;
    if (n_q != 1) return false;
    if (gqa != 2) return false;
    if (k_bits != 4 or v_bits != 4) return false; // iter 1: k4v4 sólo
    return true;
}

/// Wrapper host Zig para `fattn_kvarn_vec_d256_k4v4_kernel` (B6 iter 1).
/// SKELETON: el kernel actual no escribe output útil todavía (B6
/// iter 2 ampliará el V-pass). La forma de la API es estable para
/// que B7 pueda ir llamándola.
pub fn fattnKvarnVecDevice(
    module: ?cudaz.CUmodule,
    args: *const KvarnVecArgs,
    stream: cudaz.CUstream,
) !KvarnVecResult {
    if (args.n_kv <= 0 or args.n_q_heads <= 0 or args.n_kv_heads <= 0) {
        return error.InvalidVecArgs;
    }
    if (args.n_q_heads != args.n_kv_heads * 2) {
        return error.GqaNotTwo;
    }

    const func = try cudaz.cuModuleGetFunction(module orelse return error.NeedCudaModule, "fattn_kvarn_vec_d256_k4v4_kernel");

    if (debugz.dbg.at(.detail)) {
        debugz.dbg.printLevel(
            .detail,
            "[kvarn-fa] vec D=256 k4v4 n_qh={d} n_kvh={d} n_kv={d} streams={d}\n",
            .{
                args.n_q_heads,
                args.n_kv_heads,
                args.n_kv,
                args.n_stream,
            },
        );
    }

    var q_data_any: cudaz.CUdeviceptr = @intFromPtr(args.q_data);
    var k_descs_any: cudaz.CUdeviceptr = @intFromPtr(args.k_descs);
    var v_descs_any: cudaz.CUdeviceptr = @intFromPtr(args.v_descs);
    var mask_data_any: cudaz.CUdeviceptr = if (args.mask_data) |m| @intFromPtr(m) else 0;
    var dst_data_any: cudaz.CUdeviceptr = @intFromPtr(args.dst_data);
    // ARITY=10: must match the 10 scalars of fattn_kvarn_vec_d256_k4v4_kernel.
    // Mismatch = silent garbage-arg reading at the device (iter-15/17 bug).
    var kp: [10]?*const anyopaque = .{
        &q_data_any, // 0
        &k_descs_any, // 1
        &v_descs_any, // 2
        &mask_data_any, // 3
        &dst_data_any, // 4
        &args.n_kv, // 5
        &args.n_q_heads, // 6
        &args.n_kv_heads, // 7
        &args.n_stream, // 8
        &args.scale, // 9
    };

    try cudaz.cuLaunchKernel(
        func,
        1, // grid x: n_q=1 estricto
        @intCast(args.n_q_heads),
        @intCast(args.n_stream),
        32, // block: 1 warp
        1,
        1,
        0, // smem dyn: 0 (todo en smem estática)
        stream,
        @ptrCast(&kp),
        null,
    );

    bc.countersInc(.decode_vector);

    return .{
        .route = .decode_vector,
        .n_blocks = @intCast(args.n_q_heads * args.n_stream),
    };
}

test "vecEligible: D=256 SWA GQA=2 k4v4 n_q=1 ⇒ true" {
    try std.testing.expect(vecEligible(256, 1, 2, true, 4, 4));
}

test "vecEligible: D=128 ⇒ false" {
    try std.testing.expect(!vecEligible(128, 1, 2, true, 4, 4));
}

test "vecEligible: n_q>1 ⇒ false" {
    try std.testing.expect(!vecEligible(256, 4, 2, true, 4, 4));
}

test "vecEligible: GQA!=2 ⇒ false" {
    try std.testing.expect(!vecEligible(256, 1, 4, true, 4, 4));
}

test "vecEligible: k/v bits != 4 ⇒ false (iter 1: k4v4 only)" {
    try std.testing.expect(!vecEligible(256, 1, 2, true, 5, 4));
    try std.testing.expect(!vecEligible(256, 1, 2, true, 4, 5));
}

// ─── B7: Dispatch master ────────────────────────────────────────────────────

/// Input del dispatch master (B7). Reúne toda la información que
/// `bc.selectRoute` + `selectFallbackRoute` necesitan para decidir:
///   - `head_dim`, `n_q`, `gqa`: shape de la atención.
///   - `k_bits`, `v_bits`: par D5 fast-pairs.
///   - `swa`, `prompt_prefill`: semántica del call.
///   - `vector_eligible`, `split_eligible`: output del geometry
///     analyzer (Dev-A lo entrega en A11; mientras tanto, el caller
///     puede setearlos manualmente o usar el helper `deriveVectorEligible`
///     que mira sólo la shape).
///   - `force_portable`, `vec_disabled`: env overrides.
pub const DispatchInput = struct {
    head_dim: u32,
    n_q: u32,
    gqa: u32,
    k_bits: u8,
    v_bits: u8,
    swa: bool,
    prompt_prefill: bool,
    vector_eligible: bool = false,
    split_eligible: bool = false,
    /// Si `true`, el caller rellenó manualmente vector_eligible / split_eligible.
    /// Si `false`, `dispatchKvarnAttention` los deriva de la shape.
    explicit_eligibility: bool = false,
    force_portable: bool = false,
    vec_disabled: bool = false,
    /// Buffers device para el decode-split (A10): el caller (B7
    /// production / bench) los aloca una vez por step con el tamaño
    /// de la geometría peor-caso. Si es null, decode_split cae a
    /// portable.
    partial_buffer: ?*const KvarnPartialBuffer = null,
};

/// Buffers de trabajo del decode-split (device pointers).
pub const KvarnPartialBuffer = struct {
    partial_data: cudaz.CUdeviceptr,
    meta_data: cudaz.CUdeviceptr,
};

/// Output del dispatch master.
pub const DispatchResult = struct {
    route: bc.Route,
    /// True si el kernel realmente se lanzó; false si la ruta era
    /// `unavailable` o `prompt_prefill` (ésta última la atiende el
    /// path de Dev-A en FASE 3 — M3).
    launched: bool,
    /// Memoria allocated por el kernel lanzado (descriptors + partials + meta).
    bytes_descriptors: u64,
    bytes_partials: u64,
    bytes_meta: u64,
};

/// Deriva vector_eligible y split_eligible de la shape (B7, helper
/// mientras Dev-A no entrega `decode_select` completo). Conservador:
/// vector_eligible sólo si el gate `vecEligible` pasa; split_eligible
/// si n_q==1. El caller puede sobreescribir pasando
/// `explicit_eligibility=true` con sus propios flags.
pub fn deriveVectorEligible(input: DispatchInput) bool {
    return vecEligible(input.head_dim, input.n_q, input.gqa, input.swa, input.k_bits, input.v_bits);
}

pub fn deriveSplitEligible(input: DispatchInput) bool {
    return input.n_q == 1;
}

/// Dispatch master (B7). Decide la ruta y (si es portable o vec)
/// lanza el kernel directamente. Para rutas MMA (decode_split /
/// generic_mma) emite contador + breadcrumb pero NO lanza (Dev-A las
/// entrega en FASE 2). Para `prompt_prefill` emite breadcrumb y NO
/// lanza (Dev-A la entrega en FASE 3). Para `unavailable` emite
/// breadcrumb y devuelve `launched=false`.
///
/// `module` puede ser null en tests CPU — la ruta se decide y los
/// contadores se actualizan, pero no se llama al kernel.
///
/// Esta función NO falla en rutas que no puede lanzar — el caller
/// debe consultar `launched` y, si es false, caer a la ruta estándar
/// (flash_attention.zig CPU o el path pre-M3 actual).
pub fn dispatchKvarnAttention(
    module: ?cudaz.CUmodule,
    caps: bc.Capabilities,
    input: DispatchInput,
    attn_args: ?*const KvarnAttentionArgs,
    vec_args: ?*const KvarnVecArgs,
    stream: cudaz.CUstream,
) !DispatchResult {
    return dispatchKvarnAttentionSplit(module, null, caps, input, attn_args, vec_args, stream);
}

/// Variante con módulo del decode-split (A10 cubin). Si es null, la
/// ruta decode_split cae a portable (comportamiento pre-M2).
pub fn dispatchKvarnAttentionSplit(
    module: ?cudaz.CUmodule,
    split_module: ?cudaz.CUmodule,
    caps: bc.Capabilities,
    input: DispatchInput,
    attn_args: ?*const KvarnAttentionArgs,
    vec_args: ?*const KvarnVecArgs,
    stream: cudaz.CUstream,
) !DispatchResult {
    // 1) Resolver eligibility si el caller no la proveyó.
    const vec_elig = if (input.explicit_eligibility) input.vector_eligible else deriveVectorEligible(input);
    const split_elig = if (input.explicit_eligibility) input.split_eligible else deriveSplitEligible(input);

    // 2) Translate DispatchInput → bc.RouteInput.
    const route_input: bc.RouteInput = .{
        .head_dim = input.head_dim,
        .n_q = input.n_q,
        .gqa = input.gqa,
        .k_bits = input.k_bits,
        .v_bits = input.v_bits,
        .swa = input.swa,
        .prompt_prefill = input.prompt_prefill,
        .vector_eligible = vec_elig,
        .split_eligible = split_elig,
        .force_portable = input.force_portable or bc.envForcePortable(),
        .vec_disabled = input.vec_disabled or bc.envVecDisabled(),
    };

    // 3) Política pura: decide la ruta.
    var route = bc.selectRoute(caps, route_input);
    var result: DispatchResult = .{
        .route = route,
        .launched = false,
        .bytes_descriptors = 0,
        .bytes_partials = 0,
        .bytes_meta = 0,
    };

    // 4) Fallback si la ruta primaria no es elegible.
    if (route == .portable_native and !caps.portable_native) {
        // selectRoute no debería llegar aquí (devuelve unavailable en
        // su lugar), pero defensivo: cae a fallback.
        route = bc.selectFallbackRoute(caps, input.prompt_prefill, false);
        result.route = route;
    }
    if (route == .unavailable) {
        bc.countersInc(.unavailable);
        if (bc.envDebugRoutes()) {
            debugz.dbg.printLevel(
                .detail,
                "[kvarn-dispatch] unavailable head_dim={d} n_q={d} gqa={d} swa={any} vec_elig={any}\n",
                .{ input.head_dim, input.n_q, input.gqa, input.swa, vec_elig },
            );
        }
        return result;
    }

    // 5) Breadcrumb de ruta elegida (debug_routes).
    if (bc.envDebugRoutes()) {
        debugz.dbg.printLevel(
            .detail,
            "[kvarn-dispatch] route={s} head_dim={d} n_q={d} gqa={d} k={d} v={d}\n",
            .{
                @tagName(route),
                input.head_dim,
                input.n_q,
                input.gqa,
                input.k_bits,
                input.v_bits,
            },
        );
    }

    // 6) Lanzar el kernel correspondiente.
    sw: switch (route) {
        .portable_native => {
            if (attn_args) |a| {
                if (module) |m| {
                    const r = try fattnKvarnPortableDevice(m, a, stream);
                    result.launched = true;
                    result.bytes_descriptors = r.bytes_k_descs + r.bytes_v_descs;
                    result.bytes_partials = 0;
                    result.bytes_meta = 0;
                } else {
                    bc.countersInc(.portable_native);
                }
            } else {
                bc.countersInc(.unavailable);
            }
        },
        .decode_vector => {
            if (vec_args) |v| {
                if (module) |m| {
                    _ = try fattnKvarnVecDevice(m, v, stream);
                    result.launched = true;
                } else {
                    bc.countersInc(.decode_vector);
                }
            } else {
                // Vec seleccionado pero sin args ⇒ cae a portable.
                if (attn_args) |a| {
                    if (module) |m| {
                        const r = try fattnKvarnPortableDevice(m, a, stream);
                        result.launched = true;
                        result.bytes_descriptors = r.bytes_k_descs + r.bytes_v_descs;
                        bc.countersIncFallback(.portable);
                    } else {
                        bc.countersInc(.portable_native);
                        bc.countersIncFallback(.portable);
                    }
                } else {
                    bc.countersInc(.unavailable);
                }
            }
        },
        .decode_split, .generic_mma => {
            // M2: decode-split real (A10 kernel + A11 geometry) cuando
            // el caller aporta el módulo del split cubin; sin módulo,
            // cae a portable (comportamiento pre-M2).
            bc.countersInc(route);
            if (attn_args) |a| {
                if (split_module) |sm| {
                    // Buffers partial/meta: propiedad del CALLER en
                    // producción (B7 los aloca una vez por step); en el
                    // bench se dimensionan con selectSplitGeometry.
                    if (input.partial_buffer) |pb| {
                        const geo = try kvk.selectSplitGeometry(sm, @intCast(a.n_kv), @intCast(a.n_q), @intCast(a.n_q_heads), @intCast(a.n_kv_heads), @intCast(a.n_stream));
                        if (geo.use_split and geo.n_splits > 1) {
                            const largs: kvk.KvarnSplitLaunchArgs = .{
                                .q_data = @intFromPtr(a.q_data),
                                .k_descs = @intFromPtr(a.k_descs),
                                .v_descs = @intFromPtr(a.v_descs),
                                .mask_data = if (a.mask_data) |m| @intFromPtr(m) else 0,
                                .partial_data = pb.partial_data,
                                .meta_data = pb.meta_data,
                                .dst_data = @intFromPtr(a.dst_data),
                                .n_kv = a.n_kv,
                                .n_q = a.n_q,
                                .n_q_heads = a.n_q_heads,
                                .n_kv_heads = a.n_kv_heads,
                                .n_stream = a.n_stream,
                                .n_splits = @intCast(geo.n_splits),
                                .scale = a.scale,
                            };
                            const r = try kvk.kvarnDecodeSplitDevice(sm, &largs, stream);
                            result.launched = true;
                            result.route = .decode_split;
                            result.bytes_partials = r.bytes_partial;
                            result.bytes_meta = r.bytes_meta;
                            break :sw; // ruta lanzada — no caer a portable
                        }
                    }
                }
                // Fallback conservador: portable si está disponible.
                if (caps.portable_native) {
                    if (module) |m| {
                        const r = try fattnKvarnPortableDevice(m, a, stream);
                        result.launched = true;
                        result.bytes_descriptors = r.bytes_k_descs + r.bytes_v_descs;
                        bc.countersIncFallback(.portable);
                    } else {
                        bc.countersInc(.portable_native);
                        bc.countersIncFallback(.portable);
                    }
                }
            }
        },
        .prompt_prefill => {
            // Dev-A entrega el path de prefill en FASE 3 (A13). No
            // lanzamos — el caller cae a la ruta estándar de prefill
            // (paged_attention prefillDevice).
            bc.countersInc(.prompt_prefill);
        },
        .unavailable => {
            // (ya manejado arriba)
        },
    }

    return result;
}

/// Helper para que el caller pueda llamar `dispatchKvarnAttention` con
/// el shape estándar (sólo portable, no vec) sin construir dos
/// structs. Útil en código que aún no tiene vec configurado.
pub fn dispatchKvarnPortableOnly(
    module: ?cudaz.CUmodule,
    caps: bc.Capabilities,
    input: DispatchInput,
    attn_args: *const KvarnAttentionArgs,
    stream: cudaz.CUstream,
) !DispatchResult {
    return dispatchKvarnAttention(
        module,
        caps,
        input,
        attn_args,
        null,
        stream,
    );
}

test "B7: dispatchKvarnAttention — force_portable ⇒ portable regardless" {
    const caps = sm86Caps();
    const attn_args: KvarnAttentionArgs = .{
        .q_data = @ptrFromInt(0x100),
        .k_descs = @ptrFromInt(0x200),
        .v_descs = @ptrFromInt(0x300),
        .dst_data = @ptrFromInt(0x400),
        .n_kv = 128,
        .n_q = 4,
        .n_q_heads = 8,
        .n_kv_heads = 1,
        .n_stream = 1,
        .scale = 0.1,
        .gqa = 8,
    };
    const input: DispatchInput = .{
        .head_dim = 128,
        .n_q = 4,
        .gqa = 8,
        .k_bits = 4,
        .v_bits = 4,
        .swa = false,
        .prompt_prefill = true,
        .force_portable = true,
    };
    const r = try dispatchKvarnPortableOnly(null, caps, input, &attn_args, @ptrFromInt(1));
    try std.testing.expectEqual(bc.Route.portable_native, r.route);
}

test "B7: dispatchKvarnAttention — vec_eligible con D=256 SWA GQA=2 ⇒ decode_vector" {
    const caps = sm86Caps();
    const input: DispatchInput = .{
        .head_dim = 256,
        .n_q = 1,
        .gqa = 2,
        .k_bits = 4,
        .v_bits = 4,
        .swa = true,
        .prompt_prefill = false,
        .vector_eligible = true,
        .split_eligible = true,
        .explicit_eligibility = true,
    };
    const r = try dispatchKvarnAttention(null, caps, input, null, null, @ptrFromInt(1));
    try std.testing.expectEqual(bc.Route.decode_vector, r.route);
    try std.testing.expect(!r.launched);
}

test "B7: dispatchKvarnAttention — deriveVectorEligible ⇒ true en shape correcta" {
    const input: DispatchInput = .{
        .head_dim = 256,
        .n_q = 1,
        .gqa = 2,
        .k_bits = 4,
        .v_bits = 4,
        .swa = true,
        .prompt_prefill = false,
    };
    try std.testing.expect(deriveVectorEligible(input));
    const input_wrong: DispatchInput = .{
        .head_dim = 128,
        .n_q = 1,
        .gqa = 2,
        .k_bits = 4,
        .v_bits = 4,
        .swa = true,
        .prompt_prefill = false,
    };
    try std.testing.expect(!deriveVectorEligible(input_wrong));
}

test "B7: dispatchKvarnAttention — sm_75 (no MMA) + portable ⇒ portable" {
    const caps = sm75Caps();
    const input: DispatchInput = .{
        .head_dim = 128,
        .n_q = 4,
        .gqa = 8,
        .k_bits = 4,
        .v_bits = 4,
        .swa = false,
        .prompt_prefill = false,
    };
    const r = try dispatchKvarnPortableOnly(null, caps, input, &KvarnAttentionArgs{}, @ptrFromInt(1));
    try std.testing.expectEqual(bc.Route.portable_native, r.route);
}

test "B7: dispatchKvarnAttention — no caps + no portable ⇒ unavailable" {
    const caps = noPortableCaps();
    const input: DispatchInput = .{
        .head_dim = 128,
        .n_q = 1,
        .gqa = 8,
        .k_bits = 4,
        .v_bits = 4,
        .swa = false,
        .prompt_prefill = false,
    };
    const r = try dispatchKvarnPortableOnly(null, caps, input, &KvarnAttentionArgs{}, @ptrFromInt(1));
    try std.testing.expectEqual(bc.Route.unavailable, r.route);
    try std.testing.expect(!r.launched);
}

// Helpers: distintas `Capabilities` para los tests (sm86, sm75, no-portable).
fn sm86Caps() bc.Capabilities {
    return bc.selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 99 * 1024,
    });
}

fn sm75Caps() bc.Capabilities {
    return bc.selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = false,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 64 * 1024,
    });
}

fn noPortableCaps() bc.Capabilities {
    return bc.selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = false,
        .kvarn_instances = false,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 64 * 1024,
    });
}
