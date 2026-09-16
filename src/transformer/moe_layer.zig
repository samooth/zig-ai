//! moe_layer — capa FFN MoE offload GPU-only (Lane E, E5).
//!
//! Pipeline por paso (bs=1 decode):
//!   1. routerTopK: logits+softmax+top-k device sobre pesos router residentes
//!      (f32/f16/bf16 pequeños, subidos una vez al init).
//!   2. MoeCacheGpu.ensureExperts (Contrato 7): ids de experto → slots|−1,
//!      plan de fetch en device con dirección estable.
//!   3. ExpertGatherer.gatherMissing: filas missing de los 3 bancos
//!      (gate/up/down) desde host pineado al slot cache VRAM en UN launch.
//!   4. Por experto seleccionado: qgemm gate/up + swiglu + down reutilizando
//!      los kernels cuantizados existentes de layer_kernels (SIN editarlos)
//!      sobre las vistas del slot cache; acumulación ponderada con axpyMul.
//!
//! Slot caches: cada capa aloca [cache_size × expert_bytes] por banco; los
//! slots provienen del pool LRU GLOBAL (cache_gpu compartido entre capas), así
//! que unificar los tres buffers en un pool físico único entre capas es una
//! optimización posterior sin cambio semántico.
//!
//! v1 fuera de grafo: syncs host explícitos entre fases. La captura llega al
//! integrar el loop decode — todos los buffers ya son graph-safe por
//! construcción (shape fija, direcciones estables).
//!
//! Breadcrumbs: MOE_DEBUG=1 + DEBUG_LEVEL. debug.zig intacto.

const std = @import("std");
const gguf = @import("gguf");
const debugz = @import("debug");
const cudaz = @import("cudaz");
const layer_kernels = @import("layer_kernels");
const moe_cuda = @import("moe_cuda");
const offload_cache = @import("offload_cache");
const gguf_moe = @import("gguf_moe");
const host_bank = @import("host_bank");
const cpu_executor = @import("moe_cpu_executor");
const expert_stream = @import("expert_streamer");

// ── Knobs híbrido (env, leídos UNA vez — debug.zig intocado) ───────────────
// ZIG_AI_HYBRID=1          activa split PCIe/CPU por fracción Q16 (Contrato 8:
//                          REQUIERE executor CPU attached; sin él se ignora).
// ZIG_AI_HYBRID_OVERLAP=0  serializa adrede para A/B del solape.
var g_hybrid: ?bool = null;

fn hybridEnabled() bool {
    if (g_hybrid == null) g_hybrid = std.c.getenv("ZIG_AI_HYBRID") != null;
    return g_hybrid.?;
}

/// fetch_frac_q16 resuelto UNA vez (Contrato 6 vía F4): benchbw.json real o
/// cap fijo 1 como default seguro.
/// 4.3' copy-once: si el caller pasa un buffer PINNED ANÓNIMO ya
/// registrado (cuMemHostAlloc UVA — fromFileCopyOnce), MoeLayer.init usa
/// sus VAs directas SIN intentar hostRegister (el buffer ya está
/// registrado; re-registrar fallaría y envenenaría el canary) y SIN
/// copyToPinned (el caller compone las escalas externas IN-PLACE una vez).
pub var g_pinned_preregistered: ?[]const u8 = null;

/// 11.2 p2-v2 (lane-e): cuando el caller activa un bundle, init toma las
/// fuentes de SUS ventanas pageable (staging clásico) — CERO copyToPinned
/// de los bancos del GGUF (el OOM del A/B 21B: 52 capas × bancos pinned
/// ≈ 9GB + bundle 8.4GB). Seteado por cli.zig/moe_bench ANTES del attach;
/// cada capa toma su ventana por índice secuencial moe-layer. El bundle
/// source expone slotFileOffset — aquí solo guardamos lo mínimo: base del
/// payload + slot_bytes + n_experts del header (cálculo de ventanas local,
/// sin importar expert_bundle desde moe_layer para no crear ciclo de módulos).
pub const BundleSourceRef = struct {
    /// Puntero al payload mapeado (mmap pageable del bundle).
    base: []const u8,
    /// slot_bytes por kind (gate, up, down).
    slot_bytes: [3]usize,
    n_experts: usize,
    n_layers: usize,
    /// Offset del payload dentro de `base` (payload_offset del header).
    payload_offset: usize,
};
pub var g_bundle_source: ?BundleSourceRef = null;
/// Contador secuencial de capas moe attachadas bajo bundle (lo consume init).
pub var g_bundle_layer_seq: usize = 0;

var g_frac: ?u32 = null;

fn resolvedFrac() u32 {
    if (g_frac == null) {
        g_frac = cpu_executor.resolveFetchFracQ16(std.heap.page_allocator, null);
        if (debugz.dbg.at(.info))
            debugz.dbg.print("[moe_layer] fetch_frac_q16={d} (hybrid={any})\n", .{ g_frac.?, hybridEnabled() });
    }
    return g_frac.?;
}

/// Mapeo GgmlType → código del qgemmKernel existente (rowstride map de
/// layer_kernels.cu): 0=q4_0, 1=q4_1, 2=q5_k, 3=q6_k(else), 4=q4_k, 5=q8_0,
/// 6=q3_k, 7=q2_k (qtypes 4..7 habilitados por lane-a @8d96212).
fn qtypeOf(dtype: gguf.GgmlType) !u32 {
    return switch (dtype) {
        .q4_0 => 0,
        .q4_1 => 1,
        .q5_k => 2,
        .q6_k => 3,
        .q4_k => 4,
        .q8_0 => 5,
        .q3_k => 6,
        .q2_k => 7,
        // lane-b @ac05be3: type 8 = IQ3_S (110B/SB256) — bancos iq-family.
        .iq3_s => 8,
        // lane-b B2.15 @fe7832b: type 9 = IQ2_S (82B/SB256).
        .iq2_s => 9,
        // lane-a @c80a730: type 16 = IQ1_M (GEMM case 16, mapping T1 final
        // 0..16 = todos los formatos con append).
        .iq1_m => 16,
        // lane-b @72cd4fb+2dbb923: types 10..15 (rowstride map verificado):
        .iq4_nl => 10,
        .mxfp4 => 11,
        .iq3_xxs => 12,
        .iq2_xxs => 13,
        .iq2_xs => 14,
        .tq2_0 => 15,
        else => error.UnsupportedExpertDtype,
    };
}

pub const MoeLayer = struct {
    spec: gguf_moe.MoeLayerSpec,
    cfg: offload_cache.Config,
    stream: cudaz.CUstream,
    lk: *layer_kernels.LayerKernels,
    cache_gpu: *moe_cuda.MoeCacheGpu,
    gatherer: *moe_cuda.ExpertGatherer,

    /// Último overflow detectado (ids originales + pesos top-k) — lo consume
    /// el flujo híbrido de F (Contrato 8, patrón FFN-completo 01:3x) cuando
    /// haya executor attached.
    last_overflow_ids: [64]i32 = undefined,
    last_overflow_w: [64]f32 = undefined,
    last_overflow_n: usize = 0,

    /// Executors CPU de lane-f (Contrato 8, patrón FFN-completo): uno para la
    /// geometría gate/up (k=n_embd,out=ff) y otro para down (k=ff,out=n_embd).
    /// Ambos no-null ⇒ híbrido activo; si no, offload puro (frac=0 forzado).
    exec_gu: ?*cpu_executor.Executor = null,
    exec_down: ?*cpu_executor.Executor = null,

    /// Scratch host del hidden (staging del submit en modo host_staging).
    scratch_hidden: []f32 = &.{},
    /// Parciales host del flujo híbrido.
    part_gate: []f32 = &.{},
    part_up: []f32 = &.{},
    /// Acumulador host del merge de overflow (pre-alocado — nota F: sin
    /// allocs en hot-path).
    part_accum: []f32 = &.{},

    /// 4.9 prefetch N+1 (opt-in MOE_PREFETCH=1): predicción de los top
    /// expertos por decode_freq de ESTA capa stageada al FetchStream al
    /// final del forward — el paso siguiente la adopta antes del ensure.
    /// (experts, slots) con el gpa que los creó (prefetch_gpa).
    prefetch_pending: ?struct { experts: []i32, slots: []i32 } = null,
    prefetch_gpa: ?std.mem.Allocator = null,

    /// Slot cache VRAM por banco [cache_size × expert_bytes].
    dst_gate: cudaz.CUdeviceptr,
    dst_up: cudaz.CUdeviceptr,
    dst_down: cudaz.CUdeviceptr,

    /// Fuentes host: VAs del mmap registrado (zero-copy) O copia pinned.
    src_gate: []u8,
    src_up: []u8,
    src_down: []u8,
    bank_registered: bool = false,
    /// 11.2 p2-v2 (lane-e): fuentes = ventanas PAGEABLE del bundle (mmap
    /// sin registrar — file-backed no registrable, driver 580). El gather
    /// device-side exige VAs DEVICEMAP ⇒ el forward fuerza staging CLÁSICO
    /// (cuMemcpyHtoDAsync desde pageable, legal) para esta capa. Requiere
    /// src_bytes_* = stride EXACTO del gather (bundle v3 homogéneo).
    bundle_classic: bool = false,
    /// Bytes por experto de las ventanas del bundle (staging v2-v3: el
    /// classicStaging lee feat_bytes del caller — este campo documenta el
    /// contrato; los feat_bytes reales llegan del spec del forward).
    bundle_eb: [3]usize = .{ 0, 0, 0 },

    /// Router residente (bytes crudos f32/f16/bf16 [n_embd×E]).
    dev_router: cudaz.CUdeviceptr,

    /// Registro pin-after-fill del mmap completo (vivo mientras la capa).
    bank_host: ?host_bank.HostBank = null,

    // scratch bs=1
    dev_weights: cudaz.CUdeviceptr, // [top_k] f32
    dev_ids: cudaz.CUdeviceptr, // [top_k] i32 (expertos entrada → slots salida)
    dev_gate_out: cudaz.CUdeviceptr, // [ff] f32
    dev_up_out: cudaz.CUdeviceptr, // [ff] f32
    dev_ff32: cudaz.CUdeviceptr, // [ff] f32 post-swiglu
    dev_down_out: cudaz.CUdeviceptr, // [n_embd] f32
    dev_acc: cudaz.CUdeviceptr, // [n_embd] f32 acumulador ponderado

    pub fn init(
        gpa: std.mem.Allocator,
        spec: gguf_moe.MoeLayerSpec,
        cfg: offload_cache.Config,
        stream: cudaz.CUstream,
        lk: *layer_kernels.LayerKernels,
        cache_gpu: *moe_cuda.MoeCacheGpu,
        gatherer: *moe_cuda.ExpertGatherer,
        /// mmap COMPLETO del GGUF (g.data de un fromFileMmap): si se pasa y
        /// el registro pin-after-fill funciona, las fuentes del gather son
        /// VAs directas del mmap (CERO copias — Contrato 5 real). Null o
        /// fallo ⇒ fallback copyToPinned (interim).
        mmap_region: ?[]const u8,
    ) !MoeLayer {
        try cudaz.ensureContext();
        // Anti-corruption: cargar el módulo MoE y sus funciones ANTES de
        // cualquier register/probe fallido — ver moe_cuda.prewarm().
        moe_cuda.prewarm();

        // Fuentes: registro pin-after-fill del mmap completo (cero copias) o
        // fallback a copia pinned. Los VAs interiores de una región registrada
        // son fuente válida para kernels (UVA+DEVICEMAP, Linux). El canary del
        // executor decide si el hostRegister es viable; si no, fallback a copia.
        var self_bank_ptr: ?host_bank.HostBank = null;
        var registered = false;
        // 4.3' copy-once: región PINNED pre-registrada por el caller ⇒ VAs
        // directas ya válidas (UVA del cuMemHostAlloc), cero re-registro.
        if (mmap_region) |region| {
            if (g_pinned_preregistered) |pre| {
                if (region.ptr == pre.ptr) {
                    registered = true;
                    debugz.dbg.printLevel(.info, "[moe_layer] copy-once: fuentes = VAs del pinned pre-registrado ({d} MB compartidos)\n", .{pre.len / 1_000_000});
                }
            }
        }
        if (mmap_region) |region| {
            if (!registered) {
                // Contrato: el canary es solo un FAST-PATH cache; la operación
                // REAL es la fuente de verdad. Un probe sintético puede pasar
                // mientras el mmap de producción falla (tamaño/flags distintos).
                // 1) canary ya fallido (cache) → directo a copyToPinned;
                // 2) canary ok (o no probado aún) → intenta el registro REAL;
                //    en fallo: envenena el canary (capas posteriores lo saltan)
                //    y cae a copyToPinned — nunca propaga el error.
                if (cpu_executor.memopsCanary()) {
                    if (host_bank.HostBank.fromFileMmapWhole(@constCast(region))) |bank| {
                        self_bank_ptr = bank;
                        registered = true;
                    } else |err| {
                        cpu_executor.poisonMemopsCanary();
                        debugz.dbg.printLevel(.info, "[moe_layer] hostRegister del mmap falló ({s}) → canary envenenado, fallback copyToPinned\n", .{@errorName(err)});
                    }
                }
            }
        }
        var src_gate: []u8 = undefined;
        var src_up: []u8 = undefined;
        var src_down: []u8 = undefined;
        // Ticket 4.4: bancos con escalas EXTERNAS (presets UD-XL, p.ej.
        // gemma-4-26B down `ffn_down_exps.scale` [n_expert] f32) NO pueden
        // servirse zero-copy del mmap — el GEMV/qgemm canónico espera escalas
        // embebidas. composeCanonical copia el banco multiplicando cada d de
        // bloque por la escala del experto. gate/up sin .scale siguen la
        // ruta normal (registered ⇒ VAs del mmap).
        const down_ext = spec.down.external_scale != null;
        // 4.3' copy-once: en el path pre-registrado el caller YA compuso las
        // escalas externas IN-PLACE (composeExternalScalesInPlace) — el down
        // del buffer es canónico y se sirve por VAs sin copia alguna.
        const copyonce_pre = registered and g_pinned_preregistered != null and
            mmap_region != null and mmap_region.?.ptr == g_pinned_preregistered.?.ptr;
        if (down_ext and !copyonce_pre) {
            debugz.dbg.printLevel(.info, "[moe_layer] capa {d}: down con escalas EXTERNAS → banco compuesto a canónico (copyToPinned forzado)\n", .{spec.layer_id});
        }
        if (g_bundle_source) |bref| {
            // 11.2 p2-v2: fuentes = ventanas del bundle (pageable, staging
            // clásico). El índice secuencial de capa moe lo cuenta el
            // caller via g_bundle_layer_seq (init lo incrementa).
            const L = g_bundle_layer_seq;
            g_bundle_layer_seq += 1;
            const E: usize = bref.n_experts;
            // Espejo de expert_bundle.slotFileOffset/kindBaseOffset (v3:
            // bancos kind-major alineados a 4096 al INICIO, stride exacto).
            // kindBaseOffset (espejo v3): cada banco empieza alineado a
            // 4096 tras el banco anterior (stride exacto dentro del banco).
            var bases: [3]usize = undefined;
            {
                var off = bref.payload_offset;
                for (0..3) |k| {
                    bases[k] = off;
                    off += bref.n_layers * E * bref.slot_bytes[k];
                }
            }
            src_gate = @constCast(bref.base[bases[0] + L * E * bref.slot_bytes[0] ..][0 .. E * bref.slot_bytes[0]]);
            src_up = @constCast(bref.base[bases[1] + L * E * bref.slot_bytes[1] ..][0 .. E * bref.slot_bytes[1]]);
            src_down = @constCast(bref.base[bases[2] + L * E * bref.slot_bytes[2] ..][0 .. E * bref.slot_bytes[2]]);
        } else if (registered) {
            src_gate = @constCast(spec.gate.bytes);
            src_up = @constCast(spec.up.bytes);
            if (down_ext and !copyonce_pre) {
                const composed = try spec.down.composeCanonical(gpa);
                defer gpa.free(composed);
                src_down = try copyToPinned(composed);
                errdefer cudaz.pinnedFree(u8, src_down);
            } else {
                src_down = @constCast(spec.down.bytes);
            }
        } else {
            src_gate = try copyToPinned(spec.gate.bytes);
            errdefer cudaz.pinnedFree(u8, src_gate);
            src_up = try copyToPinned(spec.up.bytes);
            errdefer cudaz.pinnedFree(u8, src_up);
            if (down_ext) {
                const composed = try spec.down.composeCanonical(gpa);
                defer gpa.free(composed);
                src_down = try copyToPinned(composed);
                errdefer cudaz.pinnedFree(u8, src_down);
            } else {
                src_down = try copyToPinned(spec.down.bytes);
                errdefer cudaz.pinnedFree(u8, src_down);
            }
        }

        const eb_gate = spec.gate.expertBytes();
        const eb_up = spec.up.expertBytes();
        const eb_down = spec.down.expertBytes();
        const csz: usize = cfg.cache_size;
        const n_embd: usize = @intCast(spec.router.n_embd);
        const ff: usize = @intCast(spec.gate.out_dim);
        const topk: usize = spec.top_k;

        // Scratch híbrido (Contrato 8): hidden host + parciales gate/up.
        const topk_us: usize = @intCast(spec.top_k);
        const scratch_hidden = try cudaz.pinnedAlloc(f32, @intCast(n_embd));
        errdefer cudaz.pinnedFree(f32, scratch_hidden);
        const part_gate = try gpaAlloc(gpa, f32, topk_us * ff);
        errdefer gpaFree(gpa, f32, part_gate);
        const part_up = try gpaAlloc(gpa, f32, topk_us * ff);
        errdefer gpaFree(gpa, f32, part_up);
        const part_accum = try gpaAlloc(gpa, f32, @intCast(n_embd));
        errdefer gpaFree(gpa, f32, part_accum);

        var self = MoeLayer{
            .bundle_classic = g_bundle_source != null,
            .bundle_eb = if (g_bundle_source) |b| b.slot_bytes else .{ 0, 0, 0 },
            .spec = spec,
            .scratch_hidden = scratch_hidden,
            .part_gate = part_gate,
            .part_up = part_up,
            .part_accum = part_accum,
            .cfg = cfg,
            .stream = stream,
            .lk = lk,
            .cache_gpu = cache_gpu,
            .gatherer = gatherer,
            .dst_gate = try cudaz.cuMemAlloc(csz * eb_gate),
            .dst_up = try cudaz.cuMemAlloc(csz * eb_up),
            .dst_down = try cudaz.cuMemAlloc(csz * eb_down),
            .src_gate = src_gate,
            .src_up = src_up,
            .src_down = src_down,
            .bank_registered = registered,
            .dev_router = try cudaz.cuMemAlloc(spec.router.bytes.len),
            .dev_weights = try cudaz.cuMemAlloc(topk * @sizeOf(f32)),
            .dev_ids = try cudaz.cuMemAlloc(topk * @sizeOf(i32)),
            .dev_gate_out = try cudaz.cuMemAlloc(ff * @sizeOf(f32)),
            .dev_up_out = try cudaz.cuMemAlloc(ff * @sizeOf(f32)),
            .dev_ff32 = try cudaz.cuMemAlloc(ff * @sizeOf(f32)),
            .dev_down_out = try cudaz.cuMemAlloc(n_embd * @sizeOf(f32)),
            .dev_acc = try cudaz.cuMemAlloc(n_embd * @sizeOf(f32)),
        };
        errdefer self.deinit(gpa);
        errdefer {
            inline for (.{ self.dst_gate, self.dst_up, self.dst_down }) |p| cudaz.cuMemFree(p);
        }

        if (self_bank_ptr) |*b| {
            self.bank_host = b.*;
        }
        try cudaz.cuMemcpyHtoD(self.dev_router, @intFromPtr(spec.router.bytes.ptr), spec.router.bytes.len);
        return self;
    }

    pub fn deinit(self: *MoeLayer, gpa: std.mem.Allocator) void {
        if (self.scratch_hidden.len > 0) cudaz.pinnedFree(f32, self.scratch_hidden);
        if (self.part_gate.len > 0) gpaFree(gpa, f32, self.part_gate);
        if (self.part_up.len > 0) gpaFree(gpa, f32, self.part_up);
        if (self.part_accum.len > 0) gpaFree(gpa, f32, self.part_accum);
        // 4.9: prefetch pendiente sin consumir (último paso) — liberar.
        if (self.prefetch_pending) |pend| {
            if (self.prefetch_gpa) |p| {
                p.free(pend.experts);
                p.free(pend.slots);
            }
            self.prefetch_pending = null;
        }
        self.scratch_hidden = &.{};
        self.part_gate = &.{};
        self.part_up = &.{};
        self.part_accum = &.{};
        self.bank_host = null;
        if (self.bank_registered and self.src_gate.len > 0) {
            // fuentes zero-copy: slices prestados del mmap — NO liberar
        } else if (self.src_gate.len > 0) {
            cudaz.pinnedFree(u8, self.src_gate);
            cudaz.pinnedFree(u8, self.src_up);
            cudaz.pinnedFree(u8, self.src_down);
        }
        self.bank_registered = false;
        self.src_gate = &.{};
        self.src_up = &.{};
        self.src_down = &.{};
        inline for (.{ self.dst_gate, self.dst_up, self.dst_down, self.dev_router, self.dev_weights, self.dev_ids, self.dev_gate_out, self.dev_up_out, self.dev_ff32, self.dev_down_out, self.dev_acc }) |p| {
            if (p != 0) cudaz.cuMemFree(p);
        }
        self.dst_gate = 0;
        self.dst_up = 0;
        self.dst_down = 0;
    }

    /// Contrato 8: adjunta el par de Executors (gate/up y down). Ambos deben
    /// estar en modo host_staging o memops según driver; el caller los crea
    /// con Executor.initFull y las geometrías (n_embd→ff) y (ff→n_embd).
    pub fn attachExecutors(self: *MoeLayer, gu: *cpu_executor.Executor, down: *cpu_executor.Executor) void {
        self.exec_gu = gu;
        self.exec_down = down;
    }

    pub fn executorAttached(self: *const MoeLayer) bool {
        return self.exec_gu != null and self.exec_down != null;
    }

    /// Mapeo dtype de banco → Format del gemv CPU de F. Null ⇒ dtype sin
    /// soporte CPU (iq-family restante, tq, mxfp4): overflow se avisa y salta.
    fn gemvFormatOf(dtype: gguf.GgmlType) ?cpu_executor.gemv_pub.Format {
        return switch (dtype) {
            .q4_0 => .q4_0,
            .q4_1 => .q4_1,
            .q5_k => .q5_k,
            .q6_k => .q6_k,
            .q4_k => .q4_k,
            .q8_0 => .q8_0,
            // 4.6: desbloquea gemma-4 IQ2_XXS (down=iq4_nl, gate/up=iq2_xxs)
            // para el executor híbrido.
            .iq4_nl => .iq4_nl,
            .iq2_xxs => .iq2_xxs,
            else => null,
        };
    }

    fn gpaAlloc(g: std.mem.Allocator, comptime T: type, n: usize) ![]T {
        return g.alloc(T, n);
    }

    fn gpaFree(g: std.mem.Allocator, comptime T: type, buf: []T) void {
        g.free(buf);
    }

    fn copyToPinned(bytes: []const u8) ![]u8 {
        const buf = try cudaz.pinnedAlloc(u8, bytes.len);
        @memcpy(buf, bytes);
        return buf;
    }

    /// FFN MoE bs=1: `x_dev`/`ffn_out_dev` son buffers device f32 ya
    /// residentes ([n_embd]); el resultado completo del MoE queda escrito en
    /// `ffn_out_dev` (el residual lo añade el caller, igual que la densa).
    pub fn forwardGPU(self: *MoeLayer, x_dev: usize, ffn_out_dev: usize) !void {
        const E: u32 = self.spec.n_expert;
        const topk: u32 = self.spec.top_k;
        const n_embd: u32 = @intCast(self.spec.router.n_embd);

        // 4.9 prefetch N+1 (opt-in MOE_PREFETCH=1): si hay un fetch de esta
        // capa pendiente en el FetchStream (stageado al final del paso
        // anterior), el bridge garantiza que aterrizó ANTES del ensure — y
        // la adopción marca los slots como residentes para que el ensure
        // los vea como hits (fase 3) sin re-fetch.
        if (prefetchEnabled()) {
            if (self.prefetch_pending) |pend| {
                if (moe_cuda.fetchStreamShared() catch null) |fs| {
                    try fs.bridgeTo(self.stream);
                    try moe_cuda.prefetchAdopt(self.cache_gpu, self.spec.layer_id, pend.experts, pend.slots);
                }
                self.prefetch_gpa.?.free(pend.experts);
                self.prefetch_gpa.?.free(pend.slots);
                self.prefetch_pending = null;
            }
        }

        // 1) Router → pesos/ids (ids aún son expertos).
        try moe_cuda.routerTopK(self.stream, x_dev, self.dev_router, self.dev_weights, self.dev_ids, n_embd, E, topk);
        try cudaz.cuStreamSynchronize(self.stream);
        // Snapshot de ids ORIGINALES antes de que ensure los reescriba: los
        // overflow (−1) necesitan saber qué experto eran para el executor CPU
        // (patrón FFN-completo de F).
        var orig_ids: [64]i32 = undefined;
        try moe_cuda.dtoh(i32, orig_ids[0..topk], self.dev_ids);
        var weights_ret: [64]f32 = undefined;
        try moe_cuda.dtoh(f32, weights_ret[0..topk], self.dev_weights);
        const k: usize = @intCast(topk);

        // 2) ensure_experts_moe: reescribe ids → slots ó −1 (Contrato 7).
        // Fracción híbrida SOLO con executor attached (salvaguarda: sin CPU
        // executor, los −1 quedarían sin computar). OVERLAP=0 no cambia el
        // split, solo el solape posterior (v1 serializado de facto).
        const frac: u32 = if (hybridEnabled() and self.executorAttached())
            resolvedFrac()
        else
            0;

        // ── 4.10: modo streaming expert-por-experto (opt-in MOE_EXPERT_STREAM=1)
        // ─────────────────────────────────────────────────────────────────
        // Ping-pong 2-slot sobre el FetchStream: el fetch del experto k+1
        // solapa con los GEMM del k. Reemplaza ensure+gather de LOTE por un
        // consumidor por-experto con slots propios (no toca el LRU del pool).
        if (expert_stream.enabled()) {
            return self.forwardGPUStreaming(x_dev, ffn_out_dev, orig_ids, weights_ret, k);
        }

        try self.forwardGPUBatchAfterRouter(x_dev, ffn_out_dev, orig_ids, k, frac);
    }

    /// Camino batch clásico (pre-4.10): ensure+gather de LOTE + GEMM por
    /// seleccionado + overflow híbrido a CPU. Se extrae del cuerpo histórico
    /// de forwardGPU para que 4.10 pueda caer aquí tras un fallo de reserva
    /// de slots (fallback seguro con cualquier geometría de pool).
    /// PRE: router YA corrió (orig_ids/weights/k son el snapshot host) y el
    /// prefetch 4.9 YA se adoptó (si procedía).
    fn forwardGPUBatchAfterRouter(
        self: *MoeLayer,
        x_dev: usize,
        ffn_out_dev: usize,
        orig_ids: [64]i32,
        k: usize,
        frac: u32,
    ) !void {
        const topk: u32 = self.spec.top_k;
        const n_embd: u32 = @intCast(self.spec.router.n_embd);
        const ff: u32 = @intCast(self.spec.gate.out_dim);
        const qt_gate = try qtypeOf(self.spec.gate.dtype);
        const qt_up = try qtypeOf(self.spec.up.dtype);
        const qt_down = try qtypeOf(self.spec.down.dtype);
        var eb: [3]usize = undefined;
        eb[0] = self.spec.gate.expertBytes();
        eb[1] = self.spec.up.expertBytes();
        eb[2] = self.spec.down.expertBytes();
        const dst_bases = [3]usize{ self.dst_gate, self.dst_up, self.dst_down };
        const src_bases = [3]usize{ @intFromPtr(self.src_gate.ptr), @intFromPtr(self.src_up.ptr), @intFromPtr(self.src_down.ptr) };

        try self.cache_gpu.ensureExperts(self.stream, self.spec.layer_id, self.dev_ids, topk, frac, self.cfg.max_fetch);

        // 3) Gather fused multi-banco (NOGATHER=1 desvía a staging clásico).
        // 11.2 p2-v2 (lane-e): fuentes pageable del bundle ⇒ el kernel
        // device-side NO puede leerlas (sin DEVICEMAP) — staging clásico
        // local (cuMemcpyHtoDAsync desde pageable) solo para esta capa.
        const classic_prev = moe_cuda.g_force_classic;
        defer moe_cuda.g_force_classic = classic_prev;
        if (self.bundle_classic) moe_cuda.g_force_classic = true;
        try self.gatherer.gatherMissing(self.stream, self.cache_gpu, &dst_bases, &src_bases, &eb, moe_cuda.kGatherBlocksPerBank);
        try cudaz.cuStreamSynchronize(self.stream);

        // 4) Expert-GEMM por seleccionado + reduce ponderado.
        // Overflow (slots −1): anotados para el executor CPU de F (Contrato 8,
        // patrón FFN-completo probado por F 01:3x). Hoy se saltan en la suma
        // GPU; su parcial lo produce submit(gate/up)→sync→swiglu-host→
        // submit(down)→axpy-merge cuando executor_attached=true.
        var weights: [64]f32 = undefined;
        var slots: [64]i32 = undefined;
        try moe_cuda.dtoh(f32, weights[0..k], self.dev_weights);
        try moe_cuda.dtoh(i32, slots[0..k], self.dev_ids);

        // Clasificar overflow preservando (id original, peso) para F.
        self.last_overflow_n = 0;
        for (0..k) |j| {
            if (slots[j] < 0 and j < orig_ids.len) {
                self.last_overflow_ids[self.last_overflow_n] = orig_ids[j];
                self.last_overflow_w[self.last_overflow_n] = weights[j];
                self.last_overflow_n += 1;
            }
        }

        try cudaz.cuMemsetD8(self.dev_acc, 0, n_embd * @sizeOf(f32));
        for (0..k) |j| {
            if (slots[j] < 0) continue; // overflow → executor CPU de lane-f
            const s: usize = @intCast(slots[j]);
            const gate_base = self.dst_gate + s * eb[0];
            const up_base = self.dst_up + s * eb[1];
            const down_base = self.dst_down + s * eb[2];
            try self.lk.qgemm(x_dev, gate_base, self.dev_gate_out, 1, n_embd, ff, qt_gate);
            try self.lk.qgemm(x_dev, up_base, self.dev_up_out, 1, n_embd, ff, qt_up);
            // swigluKernel escribe IN-PLACE sobre su primer argumento.
            try self.lk.swiglu(self.dev_gate_out, self.dev_up_out, ff);
            try self.lk.qgemm(self.dev_gate_out, down_base, self.dev_down_out, 1, ff, n_embd, qt_down);
            try moe_cuda.axpyMul(self.stream, self.dev_down_out, self.dev_acc, weights[j], n_embd);
        }
        try cudaz.cuStreamSynchronize(self.stream);

        // ── Overflow híbrido (Contrato 8 — patrón FFN-completo de F) ────────
        // Los ids −1 se computan en CPU: submit(gate)+submit(up) en vuelo →
        // sync×2 → swiglu HOST in-place → submit(down) → sync → axpy ponderado.
        if (self.executorAttached() and self.last_overflow_n > 0) ov: {
            // NOTA: los Executors deben crearse con Config.fmt == formato de
            // los bancos (gemvFormatOf); si el dtype no tiene soporte CPU,
            // el caller no debe attachar (executorAttached=false ⇒ offload).

            // hidden al host (staging del submit en modo host_staging).
            try cudaz.cuMemcpyDtoH(@intFromPtr(self.scratch_hidden.ptr), x_dev, n_embd * @sizeOf(f32));

            const ids_ov = self.last_overflow_ids[0..self.last_overflow_n];
            const bank_g: []const f32 = @as([*]const f32, @ptrCast(@alignCast(self.src_gate.ptr)))[0 .. self.src_gate.len / 4];
            const bank_u: []const f32 = @as([*]const f32, @ptrCast(@alignCast(self.src_up.ptr)))[0 .. self.src_up.len / 4];
            const bank_d: []const f32 = @as([*]const f32, @ptrCast(@alignCast(self.src_down.ptr)))[0 .. self.src_down.len / 4];

            const pg = self.exec_gu.?.submit(self.spec.layer_id, @intFromPtr(self.scratch_hidden.ptr), bank_g, ids_ov) catch break :ov;
            const pu = self.exec_gu.?.submit(self.spec.layer_id, @intFromPtr(self.scratch_hidden.ptr), bank_u, ids_ov) catch {
                // 4.1: sin este drain, pg quedaría eternamente en `staged`
                // (spam de watchdog) y su slot jamás se reciclaría.
                self.exec_gu.?.drainSlot(pg);
                break :ov;
            };
            var g_part: []f32 = self.exec_gu.?.sync(pg) catch break :ov;
            defer self.exec_gu.?.freePartial(g_part);
            var u_part: []f32 = self.exec_gu.?.sync(pu) catch {
                // 4.1: pu ya computado pero no consumido → reciclar SIEMPRE;
                // g_part lo libera el defer (ownership del caller).
                self.exec_gu.?.drainSlot(pu);
                break :ov;
            };
            defer self.exec_gu.?.freePartial(u_part);

            // swiglu HOST in-place sobre gate (t filas × ff cols).
            const t: usize = ids_ov.len;
            for (0..t) |r| {
                for (0..ff) |c| {
                    const gv = g_part[r * ff + c];
                    g_part[r * ff + c] = gv / (1.0 + std.math.exp(-gv)) * u_part[r * ff + c];
                }
            }

            // u_part is no longer needed after swiglu: free early.
            const u_bytes: usize = u_part.len * @sizeOf(f32);
            self.exec_gu.?.freePartial(u_part);
            u_part = &.{};
            if (std.c.getenv("ZIG_AI_HYBRID_DEBUG") != null and debugz.dbg.at(.detail))
                debugz.dbg.printLevel(.detail, "[moe_layer] capa={d} freed u_part bytes={d}\n", .{ self.spec.layer_id, u_bytes });

            const pd = self.exec_down.?.submit(self.spec.layer_id, @intFromPtr(g_part.ptr), bank_d, ids_ov) catch break :ov;
            var d_part: []f32 = self.exec_down.?.sync(pd) catch {
                // 4.1: pd computado pero no consumido → reciclar.
                self.exec_down.?.drainSlot(pd);
                break :ov;
            };
            defer self.exec_down.?.freePartial(d_part);

            // Acumular ponderado (host-accum PRE-ALOCADO → htod → axpyMul α=1).
            const nacc: usize = @intCast(n_embd);
            const accum = self.part_accum;
            @memset(accum, 0);
            for (ids_ov, 0..) |_, j2| {
                const wj = self.last_overflow_w[j2];
                if (wj == 0) continue;
                for (0..nacc) |c2| accum[c2] += wj * d_part[j2 * nacc + c2];
            }
            try moe_cuda.htod(f32, self.dev_down_out, accum);
            try moe_cuda.axpyMul(self.stream, self.dev_down_out, self.dev_acc, 1.0, n_embd);
            try cudaz.cuStreamSynchronize(self.stream);

            // d_part is consumed; free early.
            const d_bytes: usize = d_part.len * @sizeOf(f32);
            self.exec_down.?.freePartial(d_part);
            d_part = &.{};
            const g_bytes: usize = g_part.len * @sizeOf(f32);
            self.exec_gu.?.freePartial(g_part);
            g_part = &.{};
            if (std.c.getenv("ZIG_AI_HYBRID_DEBUG") != null and debugz.dbg.at(.detail))
                debugz.dbg.printLevel(.detail, "[moe_layer] capa={d} freed d_part+g_part bytes={d}\n", .{ self.spec.layer_id, d_bytes + g_bytes });
        }

        try moe_cuda.copyF32(self.stream, self.dev_acc, ffn_out_dev, n_embd);

        // 4.9: stage del prefetch de ESTA capa para el paso siguiente —
        // top predict por decode_freq, memcpys al FetchStream (solapa con
        // el compute de las capas siguientes de ESTE paso). El adopt+bridge
        // lo hace el inicio del forward del próximo paso.
        if (prefetchEnabled()) {
            if (moe_cuda.fetchStreamShared() catch null) |fs| {
                const gpa = self.prefetch_gpa orelse std.heap.page_allocator;
                const k_pred: u32 = @min(self.cfg.max_fetch, self.cfg.cache_size);
                if (moe_cuda.prefetchPredict(gpa, self.cache_gpu, self.spec.layer_id, k_pred) catch null) |pred| {
                    if (pred.experts.len > 0) {
                        try moe_cuda.prefetchStage(fs, self.cache_gpu, self.spec.layer_id, pred.experts, pred.slots, &dst_bases, &src_bases, &eb);
                        try fs.markReady();
                        self.prefetch_pending = .{ .experts = pred.experts, .slots = pred.slots };
                        self.prefetch_gpa = gpa;
                    } else {
                        gpa.free(pred.experts);
                        gpa.free(pred.slots);
                    }
                }
            }
        }

        if (std.c.getenv("MOE_DEBUG") != null and debugz.dbg.at(.detail))
            debugz.dbg.print("[moe_layer {d}] topk={d} ff={d} overflow={d}\n", .{ self.spec.layer_id, topk, ff, self.last_overflow_n });
    }

    /// 4.10: forward MoE con streaming expert-por-experto (ping-pong 2-slot
    /// sobre FetchStream). El fetch del experto j+1 solapa con los GEMM del
    /// experto j. `orig_ids`/`weights` vienen del router (snapshot host).
    /// No usa el LRU del pool: los 2 slots del ping-pong se reservan via
    /// ensure (marcados residentes) y el gather de lote se sustituye por el
    /// consumidor por-experto.
    fn forwardGPUStreaming(
        self: *MoeLayer,
        x_dev: usize,
        ffn_out_dev: usize,
        orig_ids: [64]i32,
        weights: [64]f32,
        k: usize,
    ) !void {
        const n_embd: u32 = @intCast(self.spec.router.n_embd);
        const ff: u32 = @intCast(self.spec.gate.out_dim);
        const qt_gate = try qtypeOf(self.spec.gate.dtype);
        const qt_up = try qtypeOf(self.spec.up.dtype);
        const qt_down = try qtypeOf(self.spec.down.dtype);
        var eb: [3]usize = undefined;
        eb[0] = self.spec.gate.expertBytes();
        eb[1] = self.spec.up.expertBytes();
        eb[2] = self.spec.down.expertBytes();
        const dst_bases = [3]usize{ self.dst_gate, self.dst_up, self.dst_down };
        const src_bases = [3]usize{ @intFromPtr(self.src_gate.ptr), @intFromPtr(self.src_up.ptr), @intFromPtr(self.src_down.ptr) };

        // Lista de trabajo: (experto, peso) — orden del router preservado.
        const Work = struct { e: i32, w: f32 };
        var work: [64]Work = undefined;
        var nw: usize = 0;
        for (orig_ids[0..k], weights[0..k]) |e, w| {
            if (e < 0) continue;
            work[nw] = .{ .e = e, .w = w };
            nw += 1;
        }
        if (nw == 0) {
            try cudaz.cuMemsetD8(ffn_out_dev, 0, n_embd * @sizeOf(f32));
            return;
        }

        // Reserva de los 2 slots del ping-pong vía ensure del pool: los
        // primeros dos expertos de la lista quedan residentes y el pool
        // devuelve sus slots (contrato 7: ids → slots).
        // GUARD de validez: si el pool no puede dar 2 slots DIFERENTES
        // (cache_size<2, max_fetch<2, ids duplicados → −1), el ping-pong
        // sería incorrecto (stage sobre slot −1/base corrupta) ⇒ caer al
        // camino batch clásico, que soporta cualquier pool.
        const take = @min(nw, 2);
        var reserve_ids = [_]i32{ 0, 0 };
        for (0..take) |i| reserve_ids[i] = work[i].e;
        try moe_cuda.htod(i32, self.dev_ids, reserve_ids[0..take]);
        // Peso dummy 1: el ensure solo usa ids (máscara activa), no pesos.
        var reserve_w = [_]f32{ 1, 1 };
        try moe_cuda.htod(f32, self.dev_weights, reserve_w[0..take]);
        try self.cache_gpu.ensureExperts(self.stream, self.spec.layer_id, self.dev_ids, @intCast(take), 0, self.cfg.max_fetch);
        try cudaz.cuStreamSynchronize(self.stream);
        var reserved: [2]i32 = undefined;
        try moe_cuda.dtoh(i32, reserved[0..take], self.dev_ids);
        const slots_ok = take == 2 and reserved[0] >= 0 and reserved[1] >= 0 and reserved[0] != reserved[1];
        if (!slots_ok) {
            if (std.c.getenv("MOE_DEBUG") != null and debugz.dbg.at(.detail))
                debugz.dbg.print("[moe_layer {d}] streaming: reserva de slots inválida ({any}) → fallback batch\n", .{ self.spec.layer_id, reserved[0..take] });
            // RESTAURAR dev_ids a los ids ORIGINALES del router: el ensure
            // de la reserva lo reescribió a slots ⇒ el batch re-ensure
            // necesita los ids de experto top-k originales.
            const topk: u32 = self.spec.top_k;
            try moe_cuda.htod(i32, self.dev_ids, orig_ids[0..topk]);
            try moe_cuda.htod(f32, self.dev_weights, weights[0..topk]);
            return self.forwardGPUBatchAfterRouter(x_dev, ffn_out_dev, orig_ids, k, 0);
        }

        const fs = try moe_cuda.fetchStreamShared();
        var streamer = try expert_stream.ExpertStreamer.init(
            fs,
            reserved[0],
            if (take > 1) reserved[1] else reserved[0],
            &dst_bases,
            &src_bases,
            &eb,
        );
        defer streamer.deinit();

        try cudaz.cuMemsetD8(self.dev_acc, 0, n_embd * @sizeOf(f32));

        // Cadencia: begin(e0) → advance promociona e0 a compute y stagea e1
        // (en vuelo) → GEMM(e0) + markComputeDone → advance espera e1, stagea
        // e2 (esperando el event del slot de e0) → GEMM(e1) …
        try streamer.begin(work[0].e);
        var slot_now = try streamer.advance(if (nw > 1) work[1].e else -1);
        var next_idx: usize = 2;

        var j: usize = 0;
        while (j < nw) : (j += 1) {
            const s: usize = @intCast(slot_now);
            const gate_base = self.dst_gate + s * eb[0];
            const up_base = self.dst_up + s * eb[1];
            const down_base = self.dst_down + s * eb[2];
            try self.lk.qgemm(x_dev, gate_base, self.dev_gate_out, 1, n_embd, ff, qt_gate);
            try self.lk.qgemm(x_dev, up_base, self.dev_up_out, 1, n_embd, ff, qt_up);
            try self.lk.swiglu(self.dev_gate_out, self.dev_up_out, ff);
            try self.lk.qgemm(self.dev_gate_out, down_base, self.dev_down_out, 1, ff, n_embd, qt_down);
            try moe_cuda.axpyMul(self.stream, self.dev_down_out, self.dev_acc, work[j].w, n_embd);
            try streamer.markComputeDone(self.stream);

            if (next_idx < nw) {
                slot_now = try streamer.advance(work[next_idx].e);
                next_idx += 1;
            } else if (j + 1 < nw) {
                slot_now = try streamer.advance(-1);
            }
        }
        try streamer.wait();
        try cudaz.cuStreamSynchronize(self.stream);
        try moe_cuda.copyF32(self.stream, self.dev_acc, ffn_out_dev, n_embd);
        if (std.c.getenv("MOE_DEBUG") != null and debugz.dbg.at(.detail))
            debugz.dbg.print("[moe_layer {d}] streaming: experts={d} (2-slot ping-pong)\n", .{ self.spec.layer_id, nw });
    }

    fn prefetchEnabled() bool {
        return std.c.getenv("MOE_PREFETCH") != null;
    }
};
