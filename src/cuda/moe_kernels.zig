//! moe_kernels — launchers CUDA del slot cache MoE (Lane E, E3).
//!
//! Envuelve `src/cuda/moe_kernels.cu` (cubin `moe_kernels.cubin`) vía la
//! CUDA Driver API, mismo patrón que layer_kernels.zig. El estado del cache
//! vive en buffers device de shape fija con direcciones estables ⇒ el kernel
//! es capturable en CUDA graph (filosofía g_decode_persistent).
//!
//! Paridad: el kernel replica bit-exacto `OffloadCache.ensureExpertsMirror`
//! (src/moe/offload_cache.zig); tests/test_moe_kernels_gpu.zig compara estado
//! completo espejo↔GPU tras cada paso.
//!
//! Breadcrumbs gated: MOE_DEBUG=1 + DEBUG_LEVEL (debug.dbg). debug.zig intacto.

const std = @import("std");
const cudaz = @import("cudaz");
const build_options = @import("build_options");
const debugz = @import("debug");
const offload_cache = @import("offload_cache");

pub const Config = offload_cache.Config;

var g_module: ?cudaz.CUmodule = null;

fn loadModule() !cudaz.CUmodule {
    if (g_module) |m| return m;
    const cubin_path = build_options.moe_cubin;
    if (cubin_path.len == 0) return error.CudaUnavailable;
    try cudaz.ensureContext();
    g_module = try cudaz.cuModuleLoad(cubin_path);
    if (moeDebugOn()) debugz.dbg.print("[moe_kernels] módulo cargado: {s}\n", .{cubin_path});
    if (debugz.dbg.dump_graph) dumpFuncs();
    return g_module.?;
}

fn dumpFuncs() void {
    const mod = g_module.?;
    for (kernel_names) |kn| {
        const f = cudaz.cuModuleGetFunction(mod, kn) catch continue;
        debugz.dbg.print("[graph] DUMP_GRAPH func {x} = {s}\n", .{ @intFromPtr(f), kn });
    }
}

const kernel_names = [_][:0]const u8{
    "copyF32Kernel",
    "axpyMulKernel",
    "ensureExpertsMoeKernel",
    "gatherMissingRowsKernel",
    "routerTopKKernel",
};
var g_funcs: [kernel_names.len]?cudaz.CUfunction = .{null} ** kernel_names.len;

fn getFunc(name: [:0]const u8) !cudaz.CUfunction {
    for (kernel_names, 0..) |kn, i| {
        if (std.mem.eql(u8, kn, name)) {
            if (g_funcs[i]) |f| return f;
            const mod = try loadModule();
            if (moeDebugOn()) debugz.dbg.print("[moe_kernels] getFunc {s} (mod=0x{x})\n", .{ name, @intFromPtr(mod) });
            const f = try cudaz.cuModuleGetFunction(mod, name);
            g_funcs[i] = f;
            return f;
        }
    }
    return error.KernelNotFound;
}

/// Eager-load del módulo y TODAS sus funciones. Regla anti-corruption:
/// en drivers rotos (hostRegister fallido + probes ext_sync que rehacen
/// cuInit/cuCtxSetCurrent), un `cuModuleGetFunction` perezoso POST-fallo
/// puede segfault DENTRO de libcuda aunque el módulo sea válido. Prewarm
/// al arranque (antes de cualquier register/probe) elimina la ventana.
pub fn prewarm() void {
    if (g_module == null) {
        _ = loadModule() catch return;
    }
    for (kernel_names) |kn| {
        _ = getFunc(kn) catch {};
    }
}

pub fn moeDebugOn() bool {
    return std.c.getenv("MOE_DEBUG") != null;
}

/// H2D genérico para tests/glue.
pub fn htod(comptime T: type, dst_dev: cudaz.CUdeviceptr, src: []const T) !void {
    try cudaz.cuMemcpyHtoD(dst_dev, @intFromPtr(src.ptr), src.len * @sizeOf(T));
}

/// D2H genérico para tests/glue.
pub fn dtoh(comptime T: type, dst: []T, src_dev: cudaz.CUdeviceptr) !void {
    try cudaz.cuMemcpyDtoH(@intFromPtr(dst.ptr), src_dev, dst.len * @sizeOf(T));
}

/// Cache MoE en device: espejo device de OffloadCache (mismo layout lógico).
/// Los tamaños son fijos por construcción; las direcciones no cambian jamás
/// durante la sesión (requisito de captura en grafo).
pub const MoeCacheGpu = struct {
    cfg: Config,

    dev_slot_for_id: cudaz.CUdeviceptr,
    dev_id_of_slot: cudaz.CUdeviceptr,
    dev_usage: cudaz.CUdeviceptr,
    dev_expert_recency: cudaz.CUdeviceptr,
    dev_active_mask: cudaz.CUdeviceptr,
    dev_evict_slots: cudaz.CUdeviceptr,
    dev_src_indices: cudaz.CUdeviceptr,
    dev_step: cudaz.CUdeviceptr,
    dev_num_indices: cudaz.CUdeviceptr,
    dev_num_missing_full: cudaz.CUdeviceptr,
    dev_scratch_usage: cudaz.CUdeviceptr,
    dev_scratch_score: cudaz.CUdeviceptr,
    dev_stat_active_layer: cudaz.CUdeviceptr,
    dev_stat_missing_layer: cudaz.CUdeviceptr,
    dev_stat_fetched_layer: cudaz.CUdeviceptr,
    dev_stat_steps_layer: cudaz.CUdeviceptr,
    dev_decode_freq: cudaz.CUdeviceptr,

    fn nLe(self: *const MoeCacheGpu) usize {
        return @as(usize, self.cfg.num_layers) * self.cfg.num_experts;
    }

    /// Aloca buffers device y sube el estado inicial limpio.
    pub fn init(cfg: Config) !MoeCacheGpu {
        try cudaz.ensureContext();
        const n_le: usize = @as(usize, cfg.num_layers) * cfg.num_experts;
        const c_sz: usize = cfg.cache_size;
        const plan: usize = @max(cfg.num_experts, c_sz);

        var self = MoeCacheGpu{
            .cfg = cfg,
            .dev_slot_for_id = try cudaz.cuMemAlloc(n_le * @sizeOf(i32)),
            .dev_id_of_slot = try cudaz.cuMemAlloc(c_sz * @sizeOf(i32)),
            .dev_usage = try cudaz.cuMemAlloc(c_sz * @sizeOf(i64)),
            .dev_expert_recency = try cudaz.cuMemAlloc(n_le * @sizeOf(i64)),
            .dev_active_mask = try cudaz.cuMemAlloc(cfg.num_experts * @sizeOf(i32)),
            .dev_evict_slots = try cudaz.cuMemAlloc(plan * @sizeOf(i32)),
            .dev_src_indices = try cudaz.cuMemAlloc(plan * @sizeOf(i32)),
            .dev_step = try cudaz.cuMemAlloc(@sizeOf(i64)),
            .dev_num_indices = try cudaz.cuMemAlloc(@sizeOf(i64)),
            .dev_num_missing_full = try cudaz.cuMemAlloc(@sizeOf(i64)),
            .dev_scratch_usage = try cudaz.cuMemAlloc(c_sz * @sizeOf(i64)),
            .dev_scratch_score = try cudaz.cuMemAlloc(cfg.num_experts * @sizeOf(i64)),
            .dev_stat_active_layer = try cudaz.cuMemAlloc(cfg.num_layers * @sizeOf(i64)),
            .dev_stat_missing_layer = try cudaz.cuMemAlloc(cfg.num_layers * @sizeOf(i64)),
            .dev_stat_fetched_layer = try cudaz.cuMemAlloc(cfg.num_layers * @sizeOf(i64)),
            .dev_stat_steps_layer = try cudaz.cuMemAlloc(cfg.num_layers * @sizeOf(i64)),
            .dev_decode_freq = try cudaz.cuMemAlloc(n_le * @sizeOf(i64)),
        };
        errdefer self.deinit();
        try self.reset();
        return self;
    }

    pub fn deinit(self: *MoeCacheGpu) void {
        inline for (.{ self.dev_slot_for_id, self.dev_id_of_slot, self.dev_usage, self.dev_expert_recency, self.dev_active_mask, self.dev_evict_slots, self.dev_src_indices, self.dev_step, self.dev_num_indices, self.dev_num_missing_full, self.dev_scratch_usage, self.dev_scratch_score }) |p| {
            cudaz.cuMemFree(p);
        }
    }

    /// Estado inicial limpio: mapas −1, usage 0, recencia −1, contadores 0.
    pub fn reset(self: *MoeCacheGpu) !void {
        const n_le = self.nLe();
        const c_sz: usize = self.cfg.cache_size;
        const nl: usize = self.cfg.num_layers;
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const al = arena.allocator();

        const minus1_i32 = try fill(al, i32, n_le, -1);
        try htod(i32, self.dev_slot_for_id, minus1_i32);
        const minus1_c = try fill(al, i32, c_sz, -1);
        try htod(i32, self.dev_id_of_slot, minus1_c);
        const zeros_c = try fill(al, i64, c_sz, 0);
        try htod(i64, self.dev_usage, zeros_c);
        const minus1_rec = try fill(al, i64, n_le, -1);
        try htod(i64, self.dev_expert_recency, minus1_rec);
        const zero_s = try fill(al, i64, 1, 0);
        try htod(i64, self.dev_step, zero_s);
        try htod(i64, self.dev_num_indices, zero_s);
        try htod(i64, self.dev_num_missing_full, zero_s);
        const zeros_l = try fill(al, i64, nl, 0);
        try htod(i64, self.dev_stat_active_layer, zeros_l);
        try htod(i64, self.dev_stat_missing_layer, zeros_l);
        try htod(i64, self.dev_stat_fetched_layer, zeros_l);
        try htod(i64, self.dev_stat_steps_layer, zeros_l);
        const zeros_le = try fill(al, i64, n_le, 0);
        try htod(i64, self.dev_decode_freq, zeros_le);
    }

    /// Contrato 7: lanza el kernel. `ids_dev` apunta a [num_active] i32 y sale
    /// REESCRITO (slot ó −1). Grid 1×1, block 1×1 (v1 single-thread: paridad
    /// bit-exacta con el espejo; ver cabecera del .cu).
    pub fn ensureExperts(
        self: *MoeCacheGpu,
        stream: cudaz.CUstream,
        layer_id: u32,
        ids_dev: usize,
        num_active: u32,
        fetch_frac_q16: u32,
        max_fetch: u32,
    ) !void {
        const func = try getFunc("ensureExpertsMoeKernel");
        var p_ids = ids_dev;
        var p_layer: c_int = @intCast(layer_id);
        var p_active: c_int = @intCast(num_active);
        var p_max_fetch: c_int = @intCast(max_fetch);
        var p_frac: c_uint = fetch_frac_q16;
        var p_e: c_int = @intCast(self.cfg.num_experts);
        var p_c: c_int = @intCast(self.cfg.cache_size);
        var kp = [_]?*anyopaque{
            &p_ids,                       &self.dev_slot_for_id,       &self.dev_id_of_slot,
            &self.dev_usage,              &self.dev_step,              &self.dev_active_mask,
            &self.dev_evict_slots,        &self.dev_src_indices,       &self.dev_num_indices,
            &self.dev_num_missing_full,   &self.dev_expert_recency,    &self.dev_scratch_usage,
            &self.dev_scratch_score,      &self.dev_stat_active_layer, &self.dev_stat_missing_layer,
            &self.dev_stat_fetched_layer, &self.dev_stat_steps_layer,  &self.dev_decode_freq,
            &p_layer,                     &p_active,                   &p_max_fetch,
            &p_frac,                      &p_e,                        &p_c,
        };
        try cudaz.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, @ptrCast(&kp), null);
        if (moeDebugOn() and debugz.dbg.at(.detail))
            debugz.dbg.print("[moe_kernels] launch capa {d}: activos={d} frac={d} cap={d}\n", .{ layer_id, num_active, fetch_frac_q16, max_fetch });
    }

    /// 4.12: resize elástico device-side (Gap 4, fase C1). Re-aloca los
    /// buffers de tamaño cache_size-dependiente con direcciones NUEVAS ⇒
    /// los CUDA graphs capturados quedan INVALIDADOS (el caller debe
    /// re-capturar; protocolo rebuild FreeToken engine.py:766-909 fase 1:
    /// destroy graphs → resize → re-capture). Sin grafo activo es seguro
    /// en cualquier punto con el stream idle.
    ///
    /// El estado superviviente (ids ≤ target tras LRU-eviction espejo del
    /// host) se conserva: los slots vivos se re-mapean a índices compactos
    /// [0..n_alive) preservando su orden relativo (estable ⇒ determinista).
    pub fn resizeDevice(self: *MoeCacheGpu, stream: cudaz.CUstream, new_size: u32) !void {
        const total: u32 = self.cfg.num_layers * self.cfg.num_experts;
        const target: u32 = @min(new_size, if (total == 0) 0 else total);
        if (target == self.cfg.cache_size) return;
        try cudaz.cuStreamSynchronize(stream);

        // Snapshot host del estado actual (id_of_slot + usage) para la
        // remap compacta: solo los primeros target slots sobreviven; los
        // ids que vivan en slots ≥ target se re-mapean a huecos compactos
        // (orden de índice = determinista).
        const old_c: usize = self.cfg.cache_size;
        const ios = try std.heap.page_allocator.alloc(i32, old_c);
        defer std.heap.page_allocator.free(ios);
        try dtoh(i32, ios, self.dev_id_of_slot);

        // Remap: victimas de cola → slots libres si el id merece quedarse
        // (LRU por usage — espejo del host resize).
        const usage = try std.heap.page_allocator.alloc(i64, old_c);
        defer std.heap.page_allocator.free(usage);
        try dtoh(i64, usage, self.dev_usage);

        // Compactación: conservar los ids con MENOR índice de slot (orden
        // estable) hasta target — con eviction LRU por usage para decidir
        // quiénes sobreviven cuando hay más vivos que target.
        var alive: usize = 0;
        for (ios) |id| {
            if (id >= 0) alive += 1;
        }
        // Victim-list ordenada por usage asc (LRU primero fuera).
        const Vic = struct { slot: usize, u: i64 };
        var vics: std.ArrayList(Vic) = .empty;
        defer vics.deinit(std.heap.page_allocator);
        for (ios, 0..) |id, c| {
            if (id >= 0) try vics.append(std.heap.page_allocator, .{ .slot = c, .u = usage[c] });
        }
        std.mem.sort(Vic, vics.items, {}, struct {
            fn lt(_: void, a: Vic, b: Vic) bool {
                return a.u < b.u;
            }
        }.lt);
        const n_evict = if (alive > target) alive - target else 0;
        var evicted = std.AutoHashMapUnmanaged(usize, void).empty;
        defer evicted.deinit(std.heap.page_allocator);
        for (vics.items[0..n_evict]) |v| {
            try evicted.put(std.heap.page_allocator, v.slot, {});
        }

        // Nuevos buffers (validate antes de destruir).
        const plan: usize = @max(self.cfg.num_experts, target);
        const new_id_of_slot = try cudaz.cuMemAlloc(target * @sizeOf(i32));
        errdefer cudaz.cuMemFree(new_id_of_slot);
        const new_usage = try cudaz.cuMemAlloc(target * @sizeOf(i64));
        errdefer cudaz.cuMemFree(new_usage);
        const new_scratch_usage = try cudaz.cuMemAlloc(target * @sizeOf(i64));
        errdefer cudaz.cuMemFree(new_scratch_usage);
        const new_evict = try cudaz.cuMemAlloc(plan * @sizeOf(i32));
        errdefer cudaz.cuMemFree(new_evict);
        const new_src = try cudaz.cuMemAlloc(plan * @sizeOf(i32));
        errdefer cudaz.cuMemFree(new_src);

        // Estado compacto host → upload.
        const h_ios = try std.heap.page_allocator.alloc(i32, target);
        defer std.heap.page_allocator.free(h_ios);
        const h_usage = try std.heap.page_allocator.alloc(i64, target);
        defer std.heap.page_allocator.free(h_usage);
        @memset(h_ios, -1);
        @memset(h_usage, 0);

        // slot_for_id remap: id → nuevo índice compacto.
        const E: usize = self.cfg.num_experts;
        const n_le: usize = self.cfg.num_layers * E;
        const sfi = try std.heap.page_allocator.alloc(i32, n_le);
        defer std.heap.page_allocator.free(sfi);
        try dtoh(i32, sfi, self.dev_slot_for_id);

        var next: usize = 0;
        for (ios, 0..) |id, c| {
            if (id < 0) continue;
            if (evicted.contains(c)) {
                // id fuera: desmapear.
                if (id < n_le) sfi[@intCast(id)] = -1;
                continue;
            }
            if (next >= target) break;
            h_ios[next] = id;
            h_usage[next] = usage[c];
            if (id >= 0 and id < n_le) sfi[@intCast(id)] = @intCast(next);
            next += 1;
        }
        // sanity: ids supervivientes caben (eviction garantiza alive−evicted ≤ target)
        if (next != alive - n_evict) return error.RemapInvariant;

        try htod(i32, new_id_of_slot, h_ios);
        try htod(i64, new_usage, h_usage);
        try htod(i32, self.dev_slot_for_id, sfi);

        // Destroy viejo + commit.
        cudaz.cuMemFree(self.dev_id_of_slot);
        cudaz.cuMemFree(self.dev_usage);
        cudaz.cuMemFree(self.dev_scratch_usage);
        cudaz.cuMemFree(self.dev_evict_slots);
        cudaz.cuMemFree(self.dev_src_indices);
        self.dev_id_of_slot = new_id_of_slot;
        self.dev_usage = new_usage;
        self.dev_scratch_usage = new_scratch_usage;
        self.dev_evict_slots = new_evict;
        self.dev_src_indices = new_src;
        self.cfg.cache_size = target;

        if (moeDebugOn() and debugz.dbg.at(.info))
            debugz.dbg.printLevel(.info, "[moe_kernels] resizeDevice: {d} → {d} slots (graphs invalidados)\n", .{ old_c, target });
    }
};

pub const RebuildResult = struct {
    resized: bool = false,
    new_cache_size: u32 = 0,
};

/// Check VRAM pressure and shrink MoE cache if needed.
/// Returns whether the cache was resized and the new size.
pub fn moeMaybeRebuildUnderPressure(
    cache_gpu: ?*MoeCacheGpu,
    decode_graph: ?*anyopaque,
    stream: cudaz.CUstream,
) RebuildResult {
    const cg = cache_gpu orelse return .{};
    _ = decode_graph;

    // Query free VRAM via cuMemGetInfo
    var free_mem: usize = 0;
    var total_mem: usize = 0;
    if (cudaz.cuMemGetInfo(&free_mem, &total_mem)) |_| {
        const free_mb = free_mem / (1024 * 1024);
        const floor_mb: usize = 512;
        if (free_mb < floor_mb) {
            const current: u32 = cg.cfg.cache_size;
            if (current > 1) {
                const target: u32 = @max(current / 4, 1);
                if (target < current) {
                    cg.resizeDevice(stream, target) catch return .{};
                    return .{ .resized = true, .new_cache_size = target };
                }
            }
        }
    } else |_| {}
    return .{};
}

fn fill(allocator: std.mem.Allocator, comptime T: type, n: usize, v: T) ![]T {
    const buf = try allocator.alloc(T, n);
    @memset(buf, v);
    return buf;
}

/// Knob A/B propio (env, leído aquí — debug.zig intacto): NOGATHER=1 fuerza
/// el path de staging clásico (memcpy por fila) en vez del gather fused.
/// A/B por proceso (bench): fuerza staging clásico sin tocar env.
pub var g_force_classic: bool = false;

pub fn noGatherEnabled() bool {
    return std.c.getenv("NOGATHER") != null or g_force_classic;
}

pub const kGatherThreads: u32 = 256;
pub const kGatherBlocksPerBank: u32 = 16; // 16×256 = 4096 hilos/banco (rodilla PCIe)

/// Gather de expertos missing: bancos host-pineados → slot cache VRAM.
///
/// `dst_bases[b]`  : base VRAM del banco b del slot cache ([plan × feat_b]).
/// `src_bases[b]`  : VA host pineado del banco b (interim pinnedAlloc con
///                   copia del mmap al setup; swap mecánico a HostBank de D2
///                   cuando aterrice — Contrato 5).
/// `feat_bytes[b]` : bytes por fila (DEBE ser múltiplo de 16).
/// Las filas a copiar viven YA en device (evict_slots/src_indices del ensure)
/// y el contador num_indices también ⇒ descriptor estable + launch graph-safe.
pub const ExpertGatherer = struct {
    n_banks: usize,
    dev_dst_ptrs: cudaz.CUdeviceptr,
    dev_src_ptrs: cudaz.CUdeviceptr,
    dev_feat_bytes: cudaz.CUdeviceptr,

    pub fn init(n_banks: usize) !ExpertGatherer {
        try cudaz.ensureContext();
        return .{
            .n_banks = n_banks,
            .dev_dst_ptrs = try cudaz.cuMemAlloc(n_banks * @sizeOf(u64)),
            .dev_src_ptrs = try cudaz.cuMemAlloc(n_banks * @sizeOf(u64)),
            .dev_feat_bytes = try cudaz.cuMemAlloc(n_banks * @sizeOf(u64)),
        };
    }

    pub fn deinit(self: *ExpertGatherer) void {
        cudaz.cuMemFree(self.dev_dst_ptrs);
        cudaz.cuMemFree(self.dev_src_ptrs);
        cudaz.cuMemFree(self.dev_feat_bytes);
    }

    fn uploadDescriptor(
        self: *ExpertGatherer,
        dst_bases: []const usize,
        src_bases: []const usize,
        feat_bytes: []const usize,
    ) !void {
        std.debug.assert(dst_bases.len == self.n_banks);
        try htod(u64, self.dev_dst_ptrs, @as([]const u64, @ptrCast(dst_bases)));
        try htod(u64, self.dev_src_ptrs, @as([]const u64, @ptrCast(src_bases)));
        try htod(u64, self.dev_feat_bytes, @as([]const u64, @ptrCast(feat_bytes)));
    }

    /// Gather fusionado (UN launch para todos los bancos). NOGATHER=1 lo
    /// desvía automáticamente al staging clásico (A/B).
    pub fn gatherMissing(
        self: *ExpertGatherer,
        stream: cudaz.CUstream,
        gpu: *MoeCacheGpu,
        dst_bases: []const usize,
        src_bases: []const usize,
        feat_bytes: []const usize,
        blocks_per_bank: u32,
    ) !void {
        if (noGatherEnabled()) {
            return self.classicStaging(stream, gpu, dst_bases, src_bases, feat_bytes);
        }
        for (feat_bytes) |fb| {
            if (fb % 16 != 0) return error.FeatBytesNotMultipleOf16;
        }
        try self.uploadDescriptor(dst_bases, src_bases, feat_bytes);
        const func = try getFunc("gatherMissingRowsKernel");
        var p_dst = self.dev_dst_ptrs;
        var p_src = self.dev_src_ptrs;
        var p_feat = self.dev_feat_bytes;
        var p_rows = gpu.dev_evict_slots;
        var p_srows = gpu.dev_src_indices;
        var p_numidx = gpu.dev_num_indices;
        var p_banks: c_int = @intCast(self.n_banks);
        var p_bpb: c_int = @intCast(blocks_per_bank);
        var kp = [_]?*anyopaque{
            &p_dst,    &p_src,   &p_feat, &p_rows, &p_srows,
            &p_numidx, &p_banks, &p_bpb,
        };
        const gx: c_uint = @intCast(blocks_per_bank * @as(u32, @intCast(self.n_banks)));
        try cudaz.cuLaunchKernel(func, gx, 1, 1, kGatherThreads, 1, 1, 0, stream, @ptrCast(&kp), null);
        if (moeDebugOn() and debugz.dbg.at(.detail))
            debugz.dbg.print("[moe_gather] launch fused: banks={d} bpb={d}\n", .{ self.n_banks, blocks_per_bank });
    }

    /// Staging clásico para A/B (NOGATHER=1): un memcpy async por fila y banco.
    /// 4.8 A2: con MOE_FETCH_ASYNC=1 los memcpys van al stream de fetch
    /// dedicado (FetchStream) + bridge al compute — el event-bridge permite
    /// que la RONDA SIGUIENTE solape (prefetch 4.9); en la ronda actual el
    /// bridge preserva el orden (equivale al camino sync).
    pub fn classicStaging(
        self: *ExpertGatherer,
        stream: cudaz.CUstream,
        gpu: *MoeCacheGpu,
        dst_bases: []const usize,
        src_bases: []const usize,
        feat_bytes: []const usize,
    ) !void {
        _ = self; // el descriptor device no se usa en el path clásico
        const n_banks = dst_bases.len;
        std.debug.assert(n_banks == src_bases.len and n_banks == feat_bytes.len);
        // Descarga contadores y plan (host ya sincronizó tras ensure en tests;
        // en runtime E5 esto vive fuera del grafo o se planifica por batch).
        var nf: [1]i64 = undefined;
        try dtoh(i64, &nf, gpu.dev_num_indices);
        if (nf[0] <= 0) return;
        const n: usize = @intCast(nf[0]);
        const ev = try std.heap.page_allocator.alloc(i32, n);
        defer std.heap.page_allocator.free(ev);
        const sr = try std.heap.page_allocator.alloc(i32, n);
        defer std.heap.page_allocator.free(sr);
        try dtoh(i32, ev[0..n], gpu.dev_evict_slots);
        try dtoh(i32, sr[0..n], gpu.dev_src_indices);

        // 4.8 A2: stream de fetch dedicado (opt-in MOE_FETCH_ASYNC=1). Sin
        // el env: camino clásico intacto (memcpys en el stream de compute).
        var fs: ?*FetchStream = null;
        if (fetchAsyncOn()) fs = fetchStreamShared() catch null;
        const target: cudaz.CUstream = if (fs) |f| f.stream else stream;
        for (0..n_banks) |b| {
            for (0..n) |i| {
                const dst_row: usize = @intCast(ev[i]);
                const src_off: usize = @as(usize, @intCast(sr[i])) * feat_bytes[b];
                try cudaz.cuMemcpyHtoDAsync(
                    @as(cudaz.CUdeviceptr, dst_bases[b] + dst_row * feat_bytes[b]),
                    src_bases[b] + src_off,
                    feat_bytes[b],
                    target,
                );
            }
        }
        if (fs) |f| {
            try f.markReady();
            try f.bridgeTo(stream);
        }
    }
};

/// 4.8 A2: FetchStream singleton compartido — classicStaging lo usa si
/// MOE_FETCH_ASYNC=1. El prefetch N+1 (4.9) tomará la ref para lanzar la
/// ronda siguiente ANTES del compute de la actual.
var g_fetch_stream: ?*FetchStream = null;

pub fn fetchStreamShared() !*FetchStream {
    if (g_fetch_stream) |fs| return fs;
    const fs = try std.heap.page_allocator.create(FetchStream);
    fs.* = try FetchStream.init();
    g_fetch_stream = fs;
    return fs;
}

/// Teardown del singleton (llaman moe-bench/deinits CUDA al salir).
pub fn fetchStreamDeinit() void {
    if (g_fetch_stream) |fs| {
        fs.deinit();
        std.heap.page_allocator.destroy(fs);
        g_fetch_stream = null;
    }
}

fn fetchAsyncOn() bool {
    return std.c.getenv("MOE_FETCH_ASYNC") != null;
}

// ── 4.9: Prefetch N+1 sobre FetchStream (Gap 5) ─────────────────────────────
//
// El decode es serial por capa (router L+1 depende de gemm L) ⇒ no hay
// routing futuro conocido. El predictor usa el HISTOGRAMA que el propio
// kernel ensure ya mantiene (decode_freq[layer×E+e]): los expertos
// frecuentes históricos de la capa L+1 se fetchean por el FetchStream
// mientras la capa L computa. El ensure real de L+1 ve los slots ya
// adoptados (hit sin fetch) — solo paga fetch el routing imprevisto.
//
// Seguridad de la adopción host-side: el estado (id_of_slot/slot_for_id/
// usage) lo toca UN solo hilo del kernel ensure, que SIEMPRE corre tras
// nuestro bridge en el stream de compute. La adopción ocurre en host
// ANTES del launch del ensure de esa capa ⇒ no hay interleave posible.

/// Top-k candidatos de prefetch para `layer_id`: expertos NO residentes
/// (slot==-1) ordenados por decode_freq desc. Devuelve (expertos, slots
/// libres elegidos) paralelos. k ≤ libres disponibles.
pub fn prefetchPredict(
    gpa: std.mem.Allocator,
    gpu: *MoeCacheGpu,
    layer_id: u32,
    k: u32,
) !struct { experts: []i32, slots: []i32 } {
    const E: usize = @intCast(gpu.cfg.num_experts);
    const C: usize = @intCast(gpu.cfg.cache_size);
    const base: usize = @as(usize, layer_id) * E;

    // Snapshot del estado del cache (slots libres + residency) y del
    // histograma de routing — 3 lecturas D2H pequeñas.
    const slot_for_id = try gpa.alloc(i32, E);
    defer gpa.free(slot_for_id);
    const id_of_slot = try gpa.alloc(i32, C);
    defer gpa.free(id_of_slot);
    @memset(slot_for_id, 0);
    @memset(id_of_slot, 0);
    try dtoh(i32, slot_for_id[0..E], gpu.dev_slot_for_id + base * @sizeOf(i32)); // solo capa layer_id
    try dtoh(i32, id_of_slot[0..C], gpu.dev_id_of_slot);
    const freqs_all = try readDecodeFreq(gpa, gpu);
    defer gpa.free(freqs_all);
    const freqs = freqs_all[base .. base + E];

    // Candidatos: no residentes, freq desc.
    const Cand = struct { e: i32, f: i64 };
    var cands: std.ArrayList(Cand) = .empty;
    defer cands.deinit(gpa);
    for (freqs, 0..) |f, e| {
        if (slot_for_id[e] == -1) try cands.append(gpa, .{ .e = @intCast(e), .f = f });
    }
    std.mem.sort(Cand, cands.items, {}, struct {
        fn lt(_: void, a: Cand, b: Cand) bool {
            return a.f > b.f;
        }
    }.lt);

    // Slots libres (id_of_slot==-1) globales — recorrer ÍNDICES (el valor
    // guarda id global base+e; -1 = vacío).
    var frees: std.ArrayList(i32) = .empty;
    defer frees.deinit(gpa);
    for (id_of_slot, 0..) |sid, idx| {
        if (sid == -1) try frees.append(gpa, @intCast(idx));
    }

    const n = @min(@min(@as(usize, k), cands.items.len), frees.items.len);
    const experts = try gpa.alloc(i32, n);
    errdefer gpa.free(experts);
    const slots = try gpa.alloc(i32, n);
    errdefer gpa.free(slots);
    for (0..n) |i| {
        experts[i] = cands.items[i].e;
        slots[i] = frees.items[i];
    }
    return .{ .experts = experts, .slots = slots };
}

/// Adopción host-side de slots prefechados: marca id_of_slot/slot_for_id/
/// usage para que el ensure de la capa los vea RESIDENTES (fase 3 hit).
/// Debe llamarse tras fetchStream.bridgeTo(compute) y ANTES del launch del
/// ensure de esa capa (orden de stream garantiza la exclusión).
pub fn prefetchAdopt(
    gpu: *MoeCacheGpu,
    layer_id: u32,
    experts: []const i32,
    slots: []const i32,
) !void {
    if (experts.len == 0) return;
    const E: usize = @intCast(gpu.cfg.num_experts);
    const C: usize = @intCast(gpu.cfg.cache_size);
    const base: usize = @as(usize, layer_id) * E;

    // Lectura-modificación-escritura de las 3 estructuras (pequeñas).
    const sfi = try std.heap.page_allocator.alloc(i32, E);
    defer std.heap.page_allocator.free(sfi);
    try dtoh(i32, sfi, gpu.dev_slot_for_id + base * @sizeOf(i32));
    const ios = try std.heap.page_allocator.alloc(i32, C);
    defer std.heap.page_allocator.free(ios);
    try dtoh(i32, ios, gpu.dev_id_of_slot);
    const us = try std.heap.page_allocator.alloc(i64, C);
    defer std.heap.page_allocator.free(us);
    try dtoh(i64, us, gpu.dev_usage);

    // step actual (para recencia del adopt) + 1: los adoptados quedan
    // "recientes" para sobrevivir a la fase-2 evict del próximo ensure.
    var stepv: [1]i64 = undefined;
    try dtoh(i64, stepv[0..], gpu.dev_step);
    const step = stepv[0] + 1;

    for (experts, slots) |e, s| {
        sfi[@intCast(e)] = s;
        ios[@intCast(s)] = @intCast(base + @as(usize, @intCast(e)));
        us[@intCast(s)] = step;
    }
    try htod(i32, gpu.dev_slot_for_id + base * @sizeOf(i32), sfi);
    try htod(i32, gpu.dev_id_of_slot, ios);
    try htod(i64, gpu.dev_usage, us);
    if (moeDebugOn() and debugz.dbg.at(.detail))
        debugz.dbg.print("[moe_prefetch] adopt capa {d}: {d} slots\n", .{ layer_id, experts.len });
}

/// Staging del prefetch al FetchStream: memcpys H2D de los expertos
/// predichos a sus slots. src_bases/feat_bytes/dst_bases idénticos al
/// gather (bancos × fila). NO marca ready — el caller decide el bridge.
pub fn prefetchStage(
    fs: *FetchStream,
    gpu: *MoeCacheGpu,
    layer_id: u32,
    experts: []const i32,
    slots: []const i32,
    dst_bases: []const usize,
    src_bases: []const usize,
    feat_bytes: []const usize,
) !void {
    _ = gpu;
    _ = layer_id;
    for (0..n_banks_min(dst_bases, src_bases, feat_bytes)) |b| {
        for (experts, slots) |e, s| {
            const src_off: usize = @as(usize, @intCast(e)) * feat_bytes[b];
            const dst: cudaz.CUdeviceptr = dst_bases[b] + @as(usize, @intCast(s)) * feat_bytes[b];
            try fs.stageHtoD(dst, src_bases[b] + src_off, feat_bytes[b]);
        }
    }
}

fn n_banks_min(a: []const usize, b: []const usize, c: []const usize) usize {
    return @min(@min(a.len, b.len), c.len);
}

/// 4.8 A2 (Gap 6): fetch de expertos en STREAM DEDICADO con event-bridge al
/// stream de compute. Hoy classicStaging/gatherMissing encolan el fetch en el
/// MISMO stream del router/GEMM ⇒ el stream-order lo serializa (B1 4.7: con
/// q*=17590 el fetch extra era 1.65× MÁS LENTO por esto). Este wrapper permite
/// al fetch del paso N+1 solaparse con el compute del paso N:
///
///   fetchStream.stageAsync(...)   // encola memcpys en s_fetch
///   fetchStream.bridgeTo(compute) // compute espera event del fetch
///
/// Patrón: exllamav3 `moe_cpu_host.py:566-600` (D2H non-blocking + flag),
/// vllm `gpu_worker.py:623-680` (stream ordering), FreeToken zero-SM
/// handshake (fase 2: cuStreamWriteValue64/cuStreamWaitValue64).
///
/// NOTA: el OVERLAP REAL requiere prefetch N+1 (4.9) desde el caller —
/// este struct es la primitiva; el uso aislado (fetch-then-bridge en el
/// mismo paso) equivale al camino sync actual, NO acelera por sí solo.
pub const FetchStream = struct {
    stream: cudaz.CUstream,
    /// Event "fetch listo": el último record señala que todos los memcpys
    /// encolados en `stream` completaron.
    ready: cudaz.CUevent,
    /// Modo compat: si falla la creación del stream dedicado (contexto
    /// saturado), degrada a "mismo stream" (bridge = no-op) — el camino
    /// clásico intacto.
    dedicated: bool,

    pub fn init() !FetchStream {
        try cudaz.ensureContext();
        // Sin fallback "stream 0": CUstream es *opaque sin default — si el
        // contexto no da un stream extra, el caller degrada a camino
        // clásico (fs=null ⇒ memcpys al stream de compute).
        const s = try cudaz.cuStreamCreate(0);
        return .{ .stream = s, .ready = try cudaz.cuEventCreate(0), .dedicated = true };
    }

    pub fn deinit(self: *FetchStream) void {
        if (self.dedicated) _ = cudaz.cuStreamDestroy(self.stream);
        cudaz.cuEventDestroy(self.ready);
    }

    /// Encola un memcpy H2D async en el stream de fetch (NO en el compute).
    pub fn stageHtoD(self: *FetchStream, dst: cudaz.CUdeviceptr, src: usize, bytes: usize) !void {
        try cudaz.cuMemcpyHtoDAsync(dst, src, bytes, self.stream);
    }

    /// Señala "fetch listo" (record del event en el stream de fetch).
    /// Llamar tras el último stageHtoD de la ronda.
    pub fn markReady(self: *FetchStream) !void {
        try cudaz.cuEventRecord(self.ready, self.stream);
    }

    /// Puente: el stream de compute NO avanza hasta que el fetch señalado
    /// esté completo (wait event). Patrones referencia: cuStreamWaitEvent
    /// con flags=0 (tras cada record).
    pub fn bridgeTo(self: *FetchStream, compute: cudaz.CUstream) !void {
        try cudaz.cuStreamWaitEvent(compute, self.ready, 0);
    }

    /// Poll no bloqueante: ¿terminó la última ronda de fetch? (para el
    /// prefetch N+1: decidir si reutilizar el fetch-stream sin sync).
    pub fn isReady(self: *FetchStream) !bool {
        return cudaz.cuEventQuery(self.ready);
    }

    /// Sync dura (debug/validación): bloquea al host hasta fetch listo.
    pub fn sync(self: *FetchStream) !void {
        try cudaz.cuEventSynchronize(self.ready);
    }
};

/// Router bs=1: logits+softmax+topk en un bloque (smem dinámico = E·4B).
pub fn routerTopK(
    stream: cudaz.CUstream,
    x_dev: usize,
    router_bytes_dev: usize,
    weights_dev: usize,
    ids_dev: usize,
    n_embd: u32,
    num_experts: u32,
    top_k: u32,
) !void {
    const func = try getFunc("routerTopKKernel");
    var p_x = x_dev;
    var p_router = router_bytes_dev;
    var p_w = weights_dev;
    var p_ids = ids_dev;
    var p_d: c_int = @intCast(n_embd);
    var p_e: c_int = @intCast(num_experts);
    var p_k: c_int = @intCast(top_k);
    var kp = [_]?*anyopaque{ &p_x, &p_router, &p_w, &p_ids, &p_d, &p_e, &p_k };
    const smem: c_uint = @intCast(num_experts * @sizeOf(f32));
    const threads: c_uint = @min(num_experts, 256);
    try cudaz.cuLaunchKernel(func, 1, 1, 1, threads, 1, 1, smem, stream, @ptrCast(&kp), null);
}

/// out += alpha·in
pub fn axpyMul(stream: cudaz.CUstream, in_dev: usize, out_dev: usize, alpha: f32, n: u32) !void {
    const func = try getFunc("axpyMulKernel");
    var p_in = in_dev;
    var p_out = out_dev;
    var p_a = alpha;
    var p_n: c_int = @intCast(n);
    var kp = [_]?*anyopaque{ &p_in, &p_out, &p_a, &p_n };
    try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, @ptrCast(&kp), null);
}

fn n_u(v: anytype) c_uint {
    return @intCast(v);
}

/// dst = src (buffers f32 device)
pub fn copyF32(stream: cudaz.CUstream, src_dev: usize, dst_dev: usize, n: u32) !void {
    const func = try getFunc("copyF32Kernel");
    var p_s = src_dev;
    var p_d = dst_dev;
    var p_n: c_int = @intCast(n);
    var kp = [_]?*anyopaque{ &p_s, &p_d, &p_n };
    try cudaz.cuLaunchKernel(func, n_u((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, @ptrCast(&kp), null);
}

/// Snapshot de stats acumuladas device (lectura única bajo demanda; los
/// contadores viven en buffers estables y el kernel solo hace +=).
pub const GpuStats = struct {
    active_layer: []i64,
    missing_layer: []i64,
    fetched_layer: []i64,
    steps_layer: []i64,

    pub fn deinit(self: *GpuStats, gpa: std.mem.Allocator) void {
        gpa.free(self.active_layer);
        gpa.free(self.missing_layer);
        gpa.free(self.fetched_layer);
        gpa.free(self.steps_layer);
    }
};

pub fn readStats(gpa: std.mem.Allocator, gpu: *MoeCacheGpu) !GpuStats {
    var s: GpuStats = undefined;
    s.active_layer = try gpa.alloc(i64, gpu.cfg.num_layers);
    s.missing_layer = try gpa.alloc(i64, gpu.cfg.num_layers);
    s.fetched_layer = try gpa.alloc(i64, gpu.cfg.num_layers);
    s.steps_layer = try gpa.alloc(i64, gpu.cfg.num_layers);
    errdefer {
        gpa.free(s.active_layer);
        gpa.free(s.missing_layer);
        gpa.free(s.fetched_layer);
        gpa.free(s.steps_layer);
    }
    try dtoh(i64, s.active_layer, gpu.dev_stat_active_layer);
    try dtoh(i64, s.missing_layer, gpu.dev_stat_missing_layer);
    try dtoh(i64, s.fetched_layer, gpu.dev_stat_fetched_layer);
    try dtoh(i64, s.steps_layer, gpu.dev_stat_steps_layer);
    return s;
}

pub fn readDecodeFreq(gpa: std.mem.Allocator, gpu: *MoeCacheGpu) ![]i64 {
    const buf = try gpa.alloc(i64, @as(usize, gpu.cfg.num_layers) * gpu.cfg.num_experts);
    errdefer gpa.free(buf);
    try dtoh(i64, buf, gpu.dev_decode_freq);
    return buf;
}

/// Cota oracle de hit-rate para `slots` residentes dado un histograma de
/// routing (frecuencias por experto): suma de las top-min(slots,n) / total.
/// Referencia de FreeToken oracle_hit_at_slots.
pub fn oracleHitAtSlots(freqs: []const i64, slots: usize) f64 {
    var total: i64 = 0;
    for (freqs) |f| total += f;
    if (total <= 0) return 0;
    const sorted = std.heap.page_allocator.dupe(i64, freqs) catch return 0;
    defer std.heap.page_allocator.free(sorted);
    std.mem.sort(i64, sorted, {}, struct {
        fn lt(_: void, a: i64, b: i64) bool {
            return a > b;
        }
    }.lt);
    const k = @min(slots, sorted.len);
    var top: i64 = 0;
    for (sorted[0..k]) |f| top += f;
    return @as(f64, @floatFromInt(top)) / @as(f64, @floatFromInt(total));
}
