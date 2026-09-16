//! offload_cache — slot cache LRU global para expertos MoE (Lane E, P2b).
//!
//! Port de `OffloadMoeCache` (FreeToken offload_cache.py:96-276) y del kernel
//! `_ensure_experts_hybrid_kernel` (offload_kernels.py:290-410) a Zig.
//!
//! Un pool ÚNICO de slots VRAM compartido por todas las capas MoE:
//!   - `slot_for_id[L*E]`   : id plano (layer·E+expert) → slot, -1 si no residente
//!   - `id_of_slot[C]`      : slot → id plano, -1 si vacío (evict sin decodificar:
//!     membership por range-check `base <= id < base+E`)
//!   - `last_access[C]`    : último clock en que el slot fue usado/asignado (LFRU)
//!   - `expert_recency[L*E]`: último paso activo por experto (prioriza misses
//!     recurrentes al elegir qué fetchar)
//!
//! TODO dinámico se decide con aritmética entera en punto fijo (Q16, cero
//! floats) para ser capturable en CUDA graph: la MISMA secuencia de operaciones
//! enteras está implementada aquí como **espejo CPU bit-exacto**
//! (`ensureExpertsMirror`) — es el oráculo de paridad del kernel device (E3) y
//! el fallback sin GPU.
//!
//! Breadcrumbs: MOE_DEBUG=1 (misses/fetches por capa), gated además por
//! DEBUG_LEVEL vía debug.dbg. debug.zig queda intocado.

const std = @import("std");
const debug = @import("debug");
const budget = @import("budget");
const disk_tier = @import("disk_tier");
const tier = @import("tier");

pub const OffloadError = error{OutOfMemory};

/// Estados de residenciá de un slot en el cache 3-tier.
pub const SlotState = enum(u8) {
    Empty = 0,
    VramResident = 1,
    CpuResident = 2,
    DiskResident = 3,
};

/// Sentinel idéntico al del kernel Triton (−2^60): score de expertos no-miss.
pub const SCORE_SENTINEL: i64 = -1152921504606846976;
/// Sentinel para slots vacíos / no-evictables.

fn moeDebug() bool {
    return std.c.getenv("MOE_DEBUG") != null;
}

pub const Config = struct {
    num_layers: u32,
    num_experts: u32,
    /// Slots totales del pool compartido (presupuesto VRAM restante).
    cache_size: u32,
    /// Cap fijo de fetches por (capa, paso) cuando fetch_frac_q16 == 0.
    max_fetch: u32 = 8,
    /// Bytes por slot (estimado). Usado para residentBytes().
    slot_bytes: u32 = 1024 * 1024,
    /// Path al FTW container para disk-backed expertos.
    disk_path: ?[]const u8 = null,
    /// FD del FTW container (-1 si no hay disk backing).
    disk_fd: i32 = -1,
    /// Ceiling de disk-resident bytes.
    disk_ceiling: usize = 0,
};

/// Cache con estado espejo en host. Los buffers device (direcciones estables
/// para captura en grafo) los añade E3 sobre este mismo layout; el espejo es
/// la referencia de verdad y el path de fallback.
pub const OffloadCache = struct {
    cfg: Config,

    // Estado residente (en GPU será buffer device con dirección estable).
    slot_for_id: []i32,
    id_of_slot: []i32,
    expert_recency: []i64,
    last_access: []u64,
    access_clock: u64 = 0,
    heat: []u32,
    active_mask: []i32,
    evict_slots: []i32,
    src_indices: []i32,
    slot_state: []SlotState,
    disk_path: []?[]const u8,
    disk_offset: []u64,
    disk_length: []usize,

    // Disk tier backend (MH-1).
    disk_tier: ?disk_tier.DiskTier,

    // Scratch del ensure (tamaños fijos, sin allocs en el hot path).
    scratch_score: []i64,
    scratch_usage: []i64,
    owner_active: []bool,

    step: i64 = 0,

    // Contadores del último ensure (equivalentes device-side).
    num_indices: i64 = 0,
    num_missing_full: i64 = 0,

    // Stats acumuladas (en el kernel real: buffers device con += capturado).
    stat_calls: i64 = 0,
    stat_active: i64 = 0,
    stat_missing: i64 = 0,
    stat_fetched: i64 = 0,
    stat_missing_layer: []i64,
    stat_active_layer: []i64,
    stat_fetched_layer: []i64,
    stat_steps_layer: []i64,

    pub fn init(allocator: std.mem.Allocator, cfg: Config) OffloadError!OffloadCache {
        if (cfg.num_layers == 0 or cfg.num_experts == 0 or cfg.cache_size == 0)
            return error.OutOfMemory;
        const le: usize = @as(usize, cfg.num_layers) * cfg.num_experts;
        const plan_slots: usize = @max(cfg.num_experts, cfg.cache_size);

        const slot_for_id = try allocFill(allocator, i32, le, -1);
        errdefer allocator.free(slot_for_id);
        const id_of_slot = try allocFill(allocator, i32, cfg.cache_size, -1);
        errdefer allocator.free(id_of_slot);
        const last_access = try allocFill(allocator, u64, cfg.cache_size, 0);
        errdefer allocator.free(last_access);
        const heat = try allocFill(allocator, u32, cfg.cache_size, 0);
        errdefer allocator.free(heat);
        const expert_recency = try allocFill(allocator, i64, le, -1);
        errdefer allocator.free(expert_recency);
        const active_mask = try allocFill(allocator, i32, cfg.num_experts, 0);
        errdefer allocator.free(active_mask);
        const evict_slots = try allocFill(allocator, i32, plan_slots, 0);
        errdefer allocator.free(evict_slots);
        const src_indices = try allocFill(allocator, i32, plan_slots, 0);
        errdefer allocator.free(src_indices);
        const scratch_score = try allocFill(allocator, i64, cfg.num_experts, 0);
        errdefer allocator.free(scratch_score);
        const scratch_usage = try allocFill(allocator, i64, cfg.cache_size, 0);
        errdefer allocator.free(scratch_usage);
        const owner_active = try allocFill(allocator, bool, cfg.cache_size, false);
        errdefer allocator.free(owner_active);
        const stat_missing_layer = try allocFill(allocator, i64, cfg.num_layers, 0);
        errdefer allocator.free(stat_missing_layer);
        const stat_active_layer = try allocFill(allocator, i64, cfg.num_layers, 0);
        errdefer allocator.free(stat_active_layer);
        const stat_fetched_layer = try allocFill(allocator, i64, cfg.num_layers, 0);
        errdefer allocator.free(stat_fetched_layer);
        const stat_steps_layer = try allocFill(allocator, i64, cfg.num_layers, 0);
        errdefer allocator.free(stat_steps_layer);
        const slot_state = try allocFill(allocator, SlotState, cfg.cache_size, .Empty);
        errdefer allocator.free(slot_state);
        const disk_path = try allocator.alloc(?[]const u8, cfg.cache_size);
        errdefer allocator.free(disk_path);
        @memset(disk_path, null);
        const disk_offset = try allocFill(allocator, u64, cfg.cache_size, 0);
        errdefer allocator.free(disk_offset);
        const disk_length = try allocFill(allocator, usize, cfg.cache_size, 0);
        errdefer allocator.free(disk_length);

        const disk_backend = if (cfg.disk_path) |dp|
            disk_tier.DiskTier.init(allocator, cfg.disk_fd, dp, cfg.disk_ceiling)
        else
            null;

        return .{
            .cfg = cfg,
            .slot_for_id = slot_for_id,
            .id_of_slot = id_of_slot,
            .last_access = last_access,
            .heat = heat,
            .expert_recency = expert_recency,
            .active_mask = active_mask,
            .evict_slots = evict_slots,
            .src_indices = src_indices,
            .slot_state = slot_state,
            .disk_path = disk_path,
            .disk_offset = disk_offset,
            .disk_length = disk_length,
            .disk_tier = disk_backend,
            .scratch_score = scratch_score,
            .scratch_usage = scratch_usage,
            .owner_active = owner_active,
            .stat_missing_layer = stat_missing_layer,
            .stat_active_layer = stat_active_layer,
            .stat_fetched_layer = stat_fetched_layer,
            .stat_steps_layer = stat_steps_layer,
        };
    }

    pub fn deinit(self: *OffloadCache, allocator: std.mem.Allocator) void {
        inline for (.{ self.slot_for_id, self.id_of_slot, self.last_access, self.heat, self.expert_recency, self.active_mask, self.evict_slots, self.src_indices, self.slot_state, self.disk_offset, self.disk_length, self.scratch_score, self.scratch_usage, self.owner_active, self.stat_missing_layer, self.stat_active_layer, self.stat_fetched_layer, self.stat_steps_layer }) |b| allocator.free(b);
        for (self.disk_path) |p| {
            if (p) |s| allocator.free(s);
        }
        allocator.free(self.disk_path);
        if (self.disk_tier) |*dt| dt.deinit();
    }

    fn allocFill(allocator: std.mem.Allocator, comptime T: type, n: usize, v: T) OffloadError![]T {
        const buf = try allocator.alloc(T, n);
        @memset(buf, v);
        return buf;
    }

    /// Reset completo (nueva secuencia / warm-up desde cero).
    pub fn reset(self: *OffloadCache) void {
        @memset(self.slot_for_id, -1);
        @memset(self.id_of_slot, -1);
        @memset(self.last_access, 0);
        @memset(self.heat, 0);
        self.access_clock = 0;
        @memset(self.expert_recency, -1);
        self.step = 0;
        self.num_indices = 0;
        self.num_missing_full = 0;
        self.resetStats();
    }

    pub fn resetStats(self: *OffloadCache) void {
        self.stat_calls = 0;
        self.stat_active = 0;
        self.stat_missing = 0;
        self.stat_fetched = 0;
        @memset(self.stat_missing_layer, 0);
        @memset(self.stat_active_layer, 0);
        @memset(self.stat_fetched_layer, 0);
        @memset(self.stat_steps_layer, 0);
    }

    /// miss_rate acumulada (auditable contra oracle_hit_at_slots en E6).
    pub fn missRate(self: *const OffloadCache) f64 {
        if (self.stat_active == 0) return 0;
        return @as(f64, @floatFromInt(self.stat_missing)) / @as(f64, @floatFromInt(self.stat_active));
    }

    /// fetch_rate: fracción de misses realmente fetcheados (audita el q* real).
    pub fn fetchRate(self: *const OffloadCache) f64 {
        if (self.stat_missing == 0) return 0;
        return @as(f64, @floatFromInt(self.stat_fetched)) / @as(f64, @floatFromInt(self.stat_missing));
    }

    /// Espejo CPU bit-exacto de `ensure_experts_moe`.
    ///
    /// `ids`: [num_active] ids de EXPERTO (layer-local) entrada; salida
    /// reescrita in-place a slot del pool ó −1 (overflow → executor CPU de F).
    /// `fetch_frac_q16`: 0 ⇒ cap fijo `cfg.max_fetch`; 1..=65536 ⇒ split por
    /// fracción con la misma fórmula entera que el kernel (Contrato 7).
    /// `disk_tier`: backend de disco para marcar slots DiskResident cuando
    /// no hay VRAM disponible.
    ///
    /// La secuencia replica EXACTAMENTE las 3 fases del kernel Triton
    /// (offload_kernels.py:329-410), incluido el orden de escaneo de
    /// argmin/argmax (primera ocurrencia) y el punto exacto de cada tienda.
    pub fn ensureExpertsMirror(
        self: *OffloadCache,
        layer_id: u32,
        ids: []i32,
        fetch_frac_q16: u32,
        disk_tier_opt: ?*disk_tier.DiskTier,
    ) void {
        const E: i64 = @intCast(self.cfg.num_experts);
        const C: usize = @intCast(self.cfg.cache_size);
        const base: i64 = @as(i64, layer_id) * E;

        self.step += 1;
        const step = self.step;

        // ── Fase 1: activos + misses ────────────────────────────────────────
        @memset(self.active_mask, 0);
        for (ids) |id| {
            if (id >= 0 and id < E) self.active_mask[@intCast(id)] = 1;
        }
        var num_missing: i64 = 0;
        for (0..self.cfg.num_experts) |e| {
            const sidx = self.slot_for_id[@intCast(base + @as(i64, @intCast(e)))];
            const is_active = self.active_mask[e] != 0;
            const is_missing = is_active and sidx == -1;
            if (is_missing) num_missing += 1;
            self.scratch_score[e] = if (is_missing)
                self.expert_recency[@intCast(base + @as(i64, @intCast(e)))] * E + (E - @as(i64, @intCast(e)) - 1)
            else
                SCORE_SENTINEL;
        }

        // Cap: fracción Q16 o fijo (misma rama que el kernel; parámetro no
        // especializado — Contrato 7).
        var max_fetch: i64 = @intCast(self.cfg.max_fetch);
        if (fetch_frac_q16 > 0) {
            const frac: i64 = @intCast(fetch_frac_q16);
            const lo = (num_missing * frac) >> 16;
            const cost_lo = @max(lo * ((1 << 16) - frac), (num_missing - lo) * frac);
            const cost_hi = @max((lo + 1) * ((1 << 16) - frac), (num_missing - lo - 1) * frac);
            max_fetch = if (cost_lo <= cost_hi) lo else lo + 1;
        }
        const num_fetch: i64 = @min(num_missing, max_fetch);

        self.num_missing_full = num_missing;
        self.num_indices = num_fetch;

        // Hits: bump last_access (LFRU clock).
        for (0..self.cfg.num_experts) |e| {
            if (self.active_mask[e] == 0) continue;
            const sidx = self.slot_for_id[@intCast(base + @as(i64, @intCast(e)))];
            if (sidx >= 0) {
                self.last_access[@intCast(sidx)] = self.access_clock;
                self.access_clock +|= 1;
            }
        }

        // Stats (acumulan siempre; lectura host diferida — E6).
        self.stat_calls += 1;
        self.stat_active += @intCast(ids.len);
        self.stat_missing += num_missing;
        self.stat_fetched += num_fetch;
        self.stat_missing_layer[layer_id] += num_missing;
        self.stat_active_layer[layer_id] += @intCast(ids.len);
        self.stat_fetched_layer[layer_id] += num_fetch;
        self.stat_steps_layer[layer_id] += 1;

        // ── Fase 2: evict via LFRU (pick_lfru) protegiendo activos; selección de
        //    misses por score estricto recencia·E+(E−1−id) ───────────────────
        if (num_fetch > 0) {
            for (0..C) |c| {
                const oid = self.id_of_slot[c];
                var owned = false;
                if (oid >= 0 and oid >= base and oid < base + E)
                    owned = self.active_mask[@intCast(oid - base)] != 0;
                self.owner_active[c] = owned;
            }

            var i: i64 = 0;
            while (i < num_fetch) : (i += 1) {
                const pick = tier.pick_lfru(
                    self.heat,
                    self.last_access,
                    self.access_clock,
                    self.owner_active,
                ) orelse break;
                const victim = pick.slot;

                const old_id = self.id_of_slot[victim];
                if (old_id >= 0) self.slot_for_id[@intCast(old_id)] = -1;

                var winner: usize = 0;
                var bs: i64 = std.math.minInt(i64);
                for (self.scratch_score, 0..) |s, e| {
                    if (s > bs) {
                        bs = s;
                        winner = e;
                    }
                }

                self.id_of_slot[victim] = @intCast(base + @as(i64, @intCast(winner)));
                self.slot_for_id[@intCast(base + @as(i64, @intCast(winner)))] = @intCast(victim);
                self.last_access[victim] = self.access_clock;
                self.access_clock +|= 1;
                self.owner_active[victim] = true;
                self.evict_slots[@intCast(i)] = @intCast(victim);
                self.src_indices[@intCast(i)] = @intCast(winner);
                self.scratch_score[winner] = SCORE_SENTINEL;
                if (disk_tier_opt) |_| {
                    if (self.disk_path[victim] != null and self.disk_length[victim] > 0) {
                        self.slot_state[victim] = .DiskResident;
                    } else {
                        self.slot_state[victim] = .VramResident;
                    }
                } else {
                    self.slot_state[victim] = .VramResident;
                }
            }
        }

        // ── Fase 3: rewrite ids → slot ó −1; bump recencia de activos ──────
        // In-place seguro: cada elemento se lee ANTES de escribirse (el kernel
        // relee expert_ids fresco aquí porque fase 1/2 no tocaban ese buffer).
        // Guard defensivo (id inválido pasa through como −1); el kernel E3
        // replicará la misma guarda.
        for (ids) |*out| {
            if (out.* >= 0 and out.* < E) {
                out.* = self.slot_for_id[@intCast(base + out.*)];
            } else {
                out.* = -1;
            }
        }
        for (0..self.cfg.num_experts) |e| {
            if (self.active_mask[e] != 0)
                self.expert_recency[@intCast(base + @as(i64, @intCast(e)))] = step;
        }

        if (moeDebug() and debug.dbg.at(.detail))
            debug.dbg.print("[moe_cache] capa {d}: activos={d} misses={d} fetch={d} step={d}\n", .{ layer_id, ids.len, num_missing, num_fetch, step });
    }

    /// Variante con disk-backed fallback: llama a `ensureExpertsMirror` y,
    /// para slots asignados a `.DiskResident`, hace pread via `disk_tier`.
    ///
    /// Precondición: `self.cfg.disk_path` y `self.disk_tier` deben estar
    /// configurados; si no, equivale a `ensureExpertsMirror`.
    pub fn ensureExperts(
        self: *OffloadCache,
        layer_id: u32,
        ids: []i32,
        fetch_frac_q16: u32,
    ) !void {
        const dt = self.disk_tier;
        try self.ensureExpertsMirror(layer_id, ids, fetch_frac_q16, dt);

        if (dt == null) return;

        const E: i64 = @intCast(self.cfg.num_experts);
        const base: i64 = @as(i64, layer_id) * E;
        for (ids) |id| {
            if (id < 0 or id >= E) continue;
            const sidx = self.slot_for_id[@intCast(base + id)];
            if (sidx < 0) continue;
            if (self.slot_state[sidx] != .DiskResident) continue;
            const expert_id = base + @as(i64, id);
            const expert_id_u32 = @as(u32, @intCast(expert_id));
            const buf = dt.?.readExpert(self.cfg.allocator, expert_id_u32) catch |e| {
                debug.dbg.print("[moe_cache] disk pread fallo expert={d}: {s}\n", .{ expert_id, @errorName(e) });
                continue;
            };
            defer dt.?.releaseExpert(buf);
            // TODO: copiar buf al slot device/host buffer.
            // Por ahora solo registra el evento; el placeholder de copy
            // lo completa E3 cuando el slot tenga buffer device estable.
            debug.dbg.printLevel(.detail, "[moe_cache] disk pread expert={d} -> {} bytes\n", .{ expert_id, buf.len });
        }
    }

    /// Total bytes currently resident (filled slots × slot_bytes).
    pub fn residentBytes(self: *const OffloadCache) usize {
        var n: usize = 0;
        for (self.id_of_slot) |id| {
            if (id >= 0) n += 1;
        }
        return n * @as(usize, self.cfg.slot_bytes);
    }

    /// Bytes that must stay resident (in‑flight layers). Currently 0.
    pub fn shrinkableFloor(self: *const OffloadCache) usize {
        _ = self;
        return 0;
    }

    /// Expose this cache as a `budget.Consumer` for the rebuild protocol.
    /// `desired` is set equal to `current` (no proactive growth); callers
    /// can adjust before calling `budget.plan()`.
    pub fn asConsumer(self: *OffloadCache, name: []const u8) budget.Consumer {
        return .{
            .name = name,
            .current = self.residentBytes(),
            .desired = self.residentBytes(),
            .shrinkable_floor = self.shrinkableFloor(),
        };
    }

    /// Evict slots until resident bytes are at or below `budget_bytes`.
    /// Uses LFRU (oldest last_access first) to pick victims.
    pub fn evictToBudget(self: *OffloadCache, budget_bytes: usize) void {
        while (self.residentBytes() > budget_bytes) {
            var victim: ?usize = null;
            var best: u64 = std.math.maxInt(u64);
            for (self.last_access, 0..) |la, c| {
                if (self.id_of_slot[c] >= 0 and la < best) {
                    best = la;
                    victim = c;
                }
            }
            const v = victim orelse break;
            const old_id = self.id_of_slot[v];
            const slot_bytes = self.cfg.slot_bytes;
            const E: u32 = self.cfg.num_experts;
            const layer: u32 = if (E == 0) 0 else @intCast(@divTrunc(@as(i64, old_id), @as(i64, E)));
            const expert: u32 = if (E == 0) 0 else @intCast(@mod(@as(i64, old_id), @as(i64, E)));
            if (moeDebug() and debug.dbg.at(.detail)) {
                debug.dbg.printLevel(
                    .detail,
                    "[moe_cache] evict slot={d} layer={d} expert={d} bytes={d} resident_after={d}\n",
                    .{ v, layer, expert, slot_bytes, self.residentBytes() - slot_bytes },
                );
            }
            if (old_id >= 0) self.slot_for_id[@intCast(old_id)] = -1;
            self.id_of_slot[v] = -1;
            self.last_access[v] = 0;
        }
    }

    /// 4.12: resize elástico del pool SIN restart (Gap 4, fase C1).
    ///
    /// Cambia `cache_size` (nº de slots) con LRU-eviction si se reduce y
    /// realoja los arrays de tamaño-dependientes. `new_size` se capa al
    /// total (num_layers × num_experts) como en `init`. Protocolo validate-
    /// then-destroy de FreeToken (engine.py:766-909): primero garantiza
    /// que el nuevo array cabe (alloc), luego destruye el viejo — un fallo
    /// de alloc deja el cache INTACTO.
    ///
    /// La versión device (MoeCacheGpu.resizeDevice) requiere re-alloc de
    /// buffers con direcciones estables ⇒ va con ventana de grafo (el
    /// caller invalida los CUDA graphs capturados); este host-side es
    /// siempre seguro.
    pub fn resize(self: *OffloadCache, allocator: std.mem.Allocator, new_size: u32) !void {
        const total: u32 = self.cfg.num_layers * self.cfg.num_experts;
        const target: u32 = @min(new_size, if (total == 0) 0 else total);
        const old: u32 = self.cfg.cache_size;

        if (target == old) return;

        // ── Shrink: elección de supervivientes por LFRU (last_access desc,
        // índice asc en ties — determinista) ANTES de tocar los arrays.
        // Los supervivientes se COMPACTAN a [0..n_alive).
        const Vic = struct { slot: usize, la: u64 };
        var vics: std.ArrayList(Vic) = .empty;
        defer vics.deinit(allocator);
        for (self.id_of_slot, 0..) |id, c| {
            if (id >= 0) try vics.append(allocator, .{ .slot = c, .la = self.last_access[c] });
        }
        // Orden por last_access DESC ⇒ los PRIMEROS son los más recientes
        // (supervivientes); la cola (last_access asc) es la que se evicta.
        std.mem.sort(Vic, vics.items, {}, struct {
            fn lt(_: void, a: Vic, b: Vic) bool {
                if (a.la != b.la) return a.la > b.la;
                return a.slot < b.slot; // tie: índice menor sobrevive (estable)
            }
        }.lt);
        const n_alive = @min(vics.items.len, target);
        const survivors = vics.items[0..n_alive];

        // ── Validate: alloc de los nuevos arrays ANTES de liberar nada.
        const plan_slots: usize = @max(self.cfg.num_experts, target);
        const new_id_of_slot = try allocFill(allocator, i32, target, -1);
        errdefer allocator.free(new_id_of_slot);
        const new_last_access = try allocFill(allocator, u64, target, 0);
        errdefer allocator.free(new_last_access);
        const new_heat = try allocFill(allocator, u32, target, 0);
        errdefer allocator.free(new_heat);
        const new_owner_active = try allocFill(allocator, bool, target, false);
        errdefer allocator.free(new_owner_active);
        const new_evict = try allocFill(allocator, i32, plan_slots, 0);
        errdefer allocator.free(new_evict);
        const new_src = try allocFill(allocator, i32, plan_slots, 0);
        errdefer allocator.free(new_src);

        // ── Commit: compactar supervivientes en orden de slot estable y
        // re-mapear slot_for_id (los evictados quedan -1).
        for (survivors, 0..) |sv, new_idx| {
            const id = self.id_of_slot[sv.slot];
            new_id_of_slot[new_idx] = id;
            new_last_access[new_idx] = self.last_access[sv.slot];
            new_heat[new_idx] = self.heat[sv.slot];
            if (id >= 0) self.slot_for_id[@intCast(id)] = @intCast(new_idx);
        }
        // Los NO supervivientes: desmapear (slot_for_id ya los apunta a un
        // índice viejo ≥ n_alive que ya no existe).
        for (vics.items[n_alive..]) |sv| {
            const id = self.id_of_slot[sv.slot];
            if (id >= 0) self.slot_for_id[@intCast(id)] = -1;
        }

        // Destroy viejo.
        allocator.free(self.id_of_slot);
        allocator.free(self.last_access);
        allocator.free(self.heat);
        allocator.free(self.owner_active);
        allocator.free(self.evict_slots);
        allocator.free(self.src_indices);

        self.id_of_slot = new_id_of_slot;
        self.last_access = new_last_access;
        self.heat = new_heat;
        self.owner_active = new_owner_active;
        self.evict_slots = new_evict;
        self.src_indices = new_src;
        self.cfg.cache_size = target;

        if (moeDebug() and debug.dbg.at(.info))
            debug.dbg.printLevel(.info, "[moe_cache] resize {d} → {d} slots (alive={d})\n", .{ old, target, n_alive });
    }
};

/// E3 (4.12 runtime): política de shrink del rebuild bajo presión VRAM.
/// Pura y determinista (testable sin GPU): dado el tamaño actual y la VRAM
/// libre en MB, devuelve el target del rebuild (patrón FreeToken
/// engine.py:766-909 fase 2 "resize caches in-place": liberar ~1/4 del
/// pool por rebuild, floor de slots). Null = no aplicar rebuild.
///
/// Contrato con MoeCacheGpu.resizeDevice: caller llama ANTES con
/// cuMemGetInfo; el resize real valida-then-destroy (fallo de alloc
/// deja el cache INTACTO — ver resize()).
pub fn rebuildTargetUnderPressure(current_size: u32, free_mb: usize, floor_mb: usize) ?u32 {
    if (free_mb >= floor_mb) return null; // sin presión: no-op
    if (current_size <= 1) return null; // floor de slots (1): ya en mínimo
    // Shrink ~1/4 por rebuild, nunca por debajo del floor 1.
    const target: u32 = @max(current_size / 4, 1);
    if (target >= current_size) return null; // u32 underflow guard
    return target;
}

test "E3 rebuildTargetUnderPressure: política de shrink determinista" {
    // Sin presión (free >= floor): no-op en cualquier tamaño.
    try @import("std").testing.expect(rebuildTargetUnderPressure(16, 600, 512) == null);
    try @import("std").testing.expect(rebuildTargetUnderPressure(1, 600, 512) == null);
    // Presión: shrink a 1/4.
    try @import("std").testing.expectEqual(@as(u32, 4), rebuildTargetUnderPressure(16, 300, 512).?);
    try @import("std").testing.expectEqual(@as(u32, 8), rebuildTargetUnderPressure(32, 100, 512).?);
    // Floor de 1 slot: tamaños chicos no bajan de 1.
    try @import("std").testing.expectEqual(@as(u32, 1), rebuildTargetUnderPressure(3, 100, 512).?);
    try @import("std").testing.expectEqual(@as(u32, 1), rebuildTargetUnderPressure(2, 0, 512).?);
    // Ya en floor: no-op.
    try @import("std").testing.expect(rebuildTargetUnderPressure(1, 100, 512) == null);
    // Floor 0 (caller sin floor): presión extrema sigue siendo safe (target≥1).
    try @import("std").testing.expectEqual(@as(u32, 1), rebuildTargetUnderPressure(2, 0, 0).?);
}
