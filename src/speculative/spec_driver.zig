//! Driver de decodificación especulativa (lane-c C2).
//!
//! REESCRITO desde cero reutilizando los módulos del motor — la versión
//! anterior duplicaba tipos (GgufContext/GpuBuffer/PagedKVCache propios) y
//! nunca compiló. Este fichero compila limpio; los caminos que requieren
//! device quedan como `error.NotImplemented` hasta C4/C6/C7.
//!
//! Arquitecturas soportadas por SpecType:
//!   .draft_mtp    — cabeza MTP embebida en el GGUF del target
//!                   (`blk.{n}.nextn.*` + `qwen35.nextn_predict_layers`),
//!                   hereda tok_embd/lm_head. Objetivo C4.
//!   .draft_dflash/.draft_dspark/.draft_dflash2 — sidecar GGUF aparte
//!                   (--model-draft) con encoder fc+norm, KV-inject y
//!                   denoise no-causal. Objetivo C6/C7.
//!
//! Contrato 3 (PLAN_MAESTRO): este driver aloca y expone el buffer de
//! logits full-vocab [n_draft × vocab] f32 device; la verificación lee
//! exactamente lo que el draft produjo.
const std = @import("std");
const gguf = @import("gguf");
const kv_cache = @import("kv_cache");
const quant_weight = @import("quant_weight");
const debugz = @import("debug");

pub const sampler = @import("sampler.zig");

/// 5.2 (lane-b1): draft-model DFlash del sidecar (kvInject + denoise CPU).
pub const dflash_draft = @import("dflash_draft.zig");
/// C7: lookup-fill (n-gramas del contexto → slots libres del verify).
pub const lookup = @import("lookup.zig");
/// C6-infra: downloader HF de sidecars (--download-dflash/dspark/dflash2).
pub const hf_download = @import("hf_download.zig");
/// 9.5: ProfitController (BeeLlama P1.1) — adaptive draft-max controller.
/// Reexportado aquí para que `specdrv.adaptive_dm.Controller` esté
/// disponible en callers (cli.zig, main.zig) sin imports adicionales.
pub const adaptive_dm = @import("adaptive_dm.zig");
/// 9.6: LoopGuard (BeeLlama P1.2) — reasoning-loop detector.
pub const loop_guard = @import("loop_guard.zig");

pub const dflash_encoder = @import("dflash_encoder.zig"); // C6.1 (5.1, lane-c)

pub const QuantFormat = kv_cache.QuantFormat;

pub const SpecError = error{
    NotImplemented,
    InvalidState,
    ModelLoadFailed,
    DraftGenerationFailed,
    VerificationFailed,
    KVCacheError,
    VocabMismatch,
    EmptyVocab,
    ScratchTooSmall,
    AllocationError,
};

/// Modos especulativos — mismo dominio que --spec-type en main.zig.
pub const SpecType = enum {
    none,
    draft_mtp,
    draft_dflash,
    draft_dspark,
    draft_dflash2,

    pub fn fromCli(s: []const u8) ?SpecType {
        const map = .{
            .{ "none", .none },
            .{ "draft-mtp", .draft_mtp },
            .{ "draft-dflash", .draft_dflash },
            .{ "draft-dspark", .draft_dspark },
            .{ "draft-dflash2", .draft_dflash2 },
        };
        inline for (map) |m| {
            if (std.mem.eql(u8, s, m[0])) return m[1];
        }
        return null;
    }
};

pub const SpecConfig = struct {
    spec_type: SpecType = .none,
    /// Tokens draft máximos por ronda (--spec-draft-n-max).
    n_max: usize = 16,
    /// Mínimo de tokens draft útiles para no desperdiciar el verify.
    n_min: usize = 1,
    /// Confianza mínima para incluir un token draft (--spec-p-min).
    p_min: f32 = 0.1,
    /// Sidecar GGUF para dflash/dspark/dflash2 (--model-draft). Vacío = MTP.
    model_draft_path: []const u8 = "",
    quant_k: QuantFormat = .fp16,
    quant_v: QuantFormat = .fp16,
    seed: u64 = 42,
};

/// Fuente de un token draft (para DUMP_SPEC y métricas).
pub const DraftSource = enum {
    drafter,
    lookup,
};

/// Traza por posición de una ronda draft→verify (gated DUMP_SPEC).
pub const StepTrace = struct {
    pos: usize,
    draft_token: u32,
    accepted: bool,
    source: DraftSource,
    /// Confianza del drafter (probabilidad softmax del token elegido).
    confidence: f32 = 0,
};

pub const AcceptResult = struct {
    /// Tokens draft aceptados (sin contar bonus).
    n_accepted: usize,
    /// Tokens totales válidos escritos en out (aceptados + bonus).
    n_total: usize,
    /// Slots rellenados por n-gram lookup (C7; 0 en MTP puro).
    lookup_filled: usize = 0,
};

pub const SpecMetrics = struct {
    total_rounds: u64 = 0,
    total_drafted: u64 = 0,
    total_accepted: u64 = 0,
    total_bonus: u64 = 0,
    /// Rechazos contados por paso (verificación secuencial v1).
    total_rejected_step: u64 = 0,
    source_drafter: u64 = 0,
    source_lookup: u64 = 0,

    // ── Métricas del ProfitController (BeeLlama 9.5, P1.1) ──────────────
    // Estas métricas las emite el decode loop tras cada `decide()`. Mientras
    // el loop no esté enchufado (C4), quedan en sus defaults y se reportan
    // como 0 / "off" / "none". El `recordProfitDecision` se llama desde
    // main.zig tras integrar el controller.
    /// `n_max` recomendado por el ProfitController (0 = shutdown).
    adaptive_depth: i32 = 0,
    /// `depth[0].cycle_ms` (baseline EWMA) en microsegundos. Útil para
    /// comparar contra `cycle_ms` del depth activo.
    adaptive_baseline_us: f64 = 0.0,
    /// Tag del `Decision` (cast a u8 para SpecMetrics plain-old-data).
    adaptive_decision: u8 = 0, // 0=pending, 1=hold, 2=demote, 3=promote_step, 4=promote_to_best, 5=shutdown

    pub fn recordRound(self: *SpecMetrics, drafted: usize, res: AcceptResult) void {
        self.total_rounds += 1;
        self.total_drafted += drafted;
        self.total_accepted += res.n_accepted;
        // El último token de res es bonus salvo que venga todo de lookup.
        if (res.n_total > res.n_accepted) self.total_bonus += 1;
    }

    /// Registra el resultado de una llamada a `ProfitController.decide()`.
    /// No hay transición de estado en SpecMetrics — solo publicación.
    pub fn recordProfitDecision(
        self: *SpecMetrics,
        recommended_n_max: i32,
        decision: @import("adaptive_dm.zig").Decision,
        baseline_cycle_ms: f32,
    ) void {
        self.adaptive_depth = recommended_n_max;
        self.adaptive_baseline_us = @as(f64, baseline_cycle_ms) * 1000.0;
        self.adaptive_decision = @intFromEnum(decision);
    }

    pub fn acceptanceRate(self: SpecMetrics) f32 {
        if (self.total_drafted == 0) return 0;
        return @as(f32, @floatFromInt(self.total_accepted)) /
            @as(f32, @floatFromInt(self.total_drafted));
    }

    pub fn tokensPerRound(self: SpecMetrics) f32 {
        if (self.total_rounds == 0) return 0;
        return @as(f32, @floatFromInt(self.total_accepted + self.total_bonus)) /
            @as(f32, @floatFromInt(self.total_rounds));
    }
};

// ─── Detección/carga de la cabeza MTP ───────────────────────────────────────

pub const MtpHeadInfo = struct {
    /// Índice blk.N donde viven los tensores nextn (== block_count del target).
    layer_idx: usize,
    predict_layers: usize,
};

/// Localiza la cabeza MTP en el GGUF del target: tensores
/// `blk.{N}.nextn.eh_proj.weight`. Con qwen35.block_count INCLUYENDO las
/// capas nextn, N = block_count − predict_layers (p.ej. 27B NEO-MTP:
/// block_count=65, predict_layers=1 ⇒ blk.64). Se prueba primero el
/// candidato derivado de la metadata y luego escaneo descendente por si
/// el GGUF no trae la key.
pub fn detectMtp(g: *const gguf.GgufFile, block_count: usize) ?MtpHeadInfo {
    if (block_count == 0) return null;
    const predict_layers = metaUsize(g, "qwen35.nextn_predict_layers", 1);

    var buf: [64]u8 = undefined;
    var candidate: usize = block_count -| @max(predict_layers, 1);
    while (candidate > 0) : (candidate -= 1) {
        const name = std.fmt.bufPrint(&buf, "blk.{d}.nextn.eh_proj.weight", .{candidate}) catch return null;
        if (g.getTensor(name) != null) {
            return .{
                .layer_idx = candidate,
                .predict_layers = if (predict_layers >= 1) predict_layers else 1,
            };
        }
        // Sólo probamos el candidato principal + un paso de seguridad: con
        // predict_layers correcto siempre acierta a la primera.
        if (predict_layers >= 1) break;
    }
    return null;
}

fn metaUsize(g: *const gguf.GgufFile, key: []const u8, default: usize) usize {
    const v = g.getMeta(key) orelse return default;
    return switch (v) {
        .uint8 => |x| x,
        .uint16 => |x| x,
        .uint32 => |x| @intCast(x),
        .int32 => |x| if (x > 0) @intCast(x) else default,
        .uint64 => |x| @intCast(x),
        .int64 => |x| if (x > 0) @intCast(x) else default,
        .string => |s| std.fmt.parseInt(usize, s, 10) catch default,
        else => default,
    };
}

// ─── Cabeza MTP: carga y primitivas (lane-c C4.1) ───────────────────────────

/// Peso cuantizado GGUF referenciando bytes mmap (sin copia hasta dequant).
pub const QWeight = quant_weight.QuantWeight;

/// Cabeza MTP estilo DeepSeek/Qwen3.5 (`graph_mtp` de llama.cpp):
///   x0 = eh_proj([ rmsnorm(emb(tok), enorm) ; rmsnorm(h_prev, hnorm) ])
///   h_new = capa_nextn(x0)          — attn+ffn estándar con KV propia
///   logits = lm_head(rmsnorm(h_new, shared_head_norm))
/// La capa nextn (blk.{N}) la monta el caller como HybridLayer normal
/// reutilizando todo el pipeline; aquí sólo las piezas que no encajan ahí.
pub const MtpHead = struct {
    layer_idx: usize,
    n_embd: usize,
    /// [2*n_embd, n_embd] cuantizado (bf16/q6_k/... según GGUF).
    eh_proj: QWeight,
    /// Gammas RMSNorm f32 (n_embd cada una).
    enorm: []f32,
    hnorm: []f32,
    shared_head_norm: []f32,

    const Self = @This();

    pub fn load(
        allocator: std.mem.Allocator,
        g: *const gguf.GgufFile,
        info: MtpHeadInfo,
        n_embd: usize,
    ) !Self {
        const S = struct {
            fn qw(a: std.mem.Allocator, file: *const gguf.GgufFile, comptime suffix: []const u8, idx: usize) !QWeight {
                const name = try std.fmt.allocPrint(a, "blk.{d}.nextn." ++ suffix, .{idx});
                defer a.free(name);
                const t = file.getTensor(name) orelse return SpecError.ModelLoadFailed;
                return QWeight.init(t, file.tensorData(t));
            }
            fn norm(a: std.mem.Allocator, file: *const gguf.GgufFile, comptime suffix: []const u8, idx: usize) ![]f32 {
                const name = try std.fmt.allocPrint(a, "blk.{d}.nextn." ++ suffix, .{idx});
                defer a.free(name);
                const t = file.getTensor(name) orelse return SpecError.ModelLoadFailed;
                const dst = try a.alloc(f32, t.numel());
                errdefer a.free(dst);
                try gguf.dequantTensor(t, file.tensorData(t), dst);
                return dst;
            }
        };

        const eh = try S.qw(allocator, g, "eh_proj.weight", info.layer_idx);
        // QuantWeight referencia mmap del GgufFile: sin ownership, sin deinit.
        const enorm = try S.norm(allocator, g, "enorm.weight", info.layer_idx);
        errdefer allocator.free(enorm);
        const hnorm = try S.norm(allocator, g, "hnorm.weight", info.layer_idx);
        errdefer allocator.free(hnorm);
        const shnorm = try S.norm(allocator, g, "shared_head_norm.weight", info.layer_idx);
        errdefer allocator.free(shnorm);

        // Sanidad de shapes
        if (enorm.len != n_embd or hnorm.len != n_embd or shnorm.len != n_embd)
            return SpecError.ModelLoadFailed;
        const dims = eh.shape();
        if (dims.len != 2 or dims[0] != 2 * n_embd) return SpecError.ModelLoadFailed;

        return .{
            .layer_idx = info.layer_idx,
            .n_embd = n_embd,
            .eh_proj = eh,
            .enorm = enorm,
            .hnorm = hnorm,
            .shared_head_norm = shnorm,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.enorm);
        allocator.free(self.hnorm);
        allocator.free(self.shared_head_norm);
        // QuantWeight referencia mmap del GgufModel: no libera bytes.
        self.* = undefined;
    }

    fn rmsnormInto(dst: []f32, x: []const f32, gamma: []const f32) void {
        // RMSNorm canónica del motor: x / sqrt(mean(x²)+eps) * gamma.
        // eps lo fija el caller vía `eps`; aquí usamos el mismo default.
        var ssq: f64 = 0;
        for (x) |v| ssq += @as(f64, v) * v;
        const inv = 1.0 / @sqrt(ssq / @as(f64, @floatFromInt(x.len)) + 1e-5);
        const inv_f32: f32 = @floatCast(inv);
        for (dst, x, gamma) |*d, v, gm| d.* = v * inv_f32 * gm;
    }

    /// Fusión de inputs del paso draft (CPU):
    ///   out = eh_proj · [ rmsnorm(emb,enorm) ; rmsnorm(h,hnorm) ]
    /// `w_f32` es eh_proj dequantizado transpuesto [n_embd × 2·n_embd]
    /// (el caller lo cachea con dequantToF32Transposed; ver test).
    pub fn fuseAndProject(
        self: *const Self,
        w_f32: []const f32,
        emb_row: []const f32,
        hidden: []const f32,
        out: []f32,
        scratch_concat: []f32,
    ) void {
        const E = self.n_embd;
        // concat = [e_norm ; h_norm] (orden de llama.cpp: e primero)
        rmsnormInto(scratch_concat[0..E], emb_row, self.enorm);
        rmsnormInto(scratch_concat[E .. 2 * E], hidden, self.hnorm);
        // out[j] = Σ_i w[j][i] * concat[i], w row-major [n_embd × 2E]
        for (out, 0..) |*oj, j| {
            const row = w_f32[j * 2 * E ..][0 .. 2 * E];
            var acc: f64 = 0;
            for (row, scratch_concat[0 .. 2 * E]) |wv, xv| acc += @as(f64, wv) * xv;
            oj.* = @floatCast(acc);
        }
    }
};

test "MtpHead.load sobre GGUF real y fuseAndProject coherente" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = std.c.getenv("GGUF_MODEL_PATH") orelse return error.SkipZigTest;
    var model = try gguf.GgufModel.load(io, std.heap.c_allocator, std.mem.span(path));
    defer model.deinit();
    const info = detectMtp(&model.file, model.config.block_count) orelse return error.SkipZigTest;

    var head = try MtpHead.load(std.heap.c_allocator, &model.file, info, model.config.embedding_length);
    defer head.deinit(std.heap.c_allocator);

    // Dequant de eh_proj para la proyección CPU de referencia.
    const E = head.n_embd;
    const w = try std.heap.c_allocator.alloc(f32, E * 2 * E);
    defer std.heap.c_allocator.free(w);
    head.eh_proj.dequantToF32Transposed(w);

    const zero = try std.heap.c_allocator.alloc(f32, E);
    defer std.heap.c_allocator.free(zero);
    @memset(zero, 0);
    const concat = try std.heap.c_allocator.alloc(f32, 2 * E);
    defer std.heap.c_allocator.free(concat);
    const out = try std.heap.c_allocator.alloc(f32, E);
    defer std.heap.c_allocator.free(out);

    // Entrada cero ⇒ proyección cero (valida el bucle/layout sin depender
    // de los valores del modelo de test).
    head.fuseAndProject(w, zero, zero, out, concat);
    for (out) |v| try std.testing.expect(v == 0);

    // Un delta unitario no debe colgar ni producir NaN.
    zero[0] = 1.0;
    head.fuseAndProject(w, zero, zero, out, concat);
    for (out) |v| try std.testing.expect(!std.math.isNan(v));
}

// ─── Driver ─────────────────────────────────────────────────────────────────

pub const SpecDriver = struct {
    allocator: std.mem.Allocator,
    config: SpecConfig,
    metrics: SpecMetrics = .{},
    rng: std.Random.DefaultPrng,

    /// Últimas trazas de verify (gated DUMP_SPEC); se trunca a trace_cap.
    traces: std.ArrayList(StepTrace),
    trace_cap: usize = 256,

    /// Buffer device de logits full-vocab [n_max+1 × vocab] f32 (Contrato 3).
    /// Se aloca en C4 vía crt.GpuBuffer al enchufar el driver al decode loop
    /// (requiere CRT init del caller). Hoy queda fuera del struct a propósito:
    /// importar cuda_runtime aquí colisiona con la copia relativa que arrastra
    /// kv_cache/gpu_dequant.zig ("file exists in two modules").
    vocab: usize = 0,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: SpecConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
            .rng = std.Random.DefaultPrng.init(config.seed),
            .traces = .empty,
        };
    }

    pub fn deinit(self: *Self) void {
        self.traces.deinit(self.allocator);
        self.* = undefined;
    }

    /// Aloca el buffer device de logits full-vocab (Contrato 3).
    /// Stub hasta C4: se implementará con crt.GpuBuffer cuando el driver se
    /// enchufe al decode loop (el caller habrá hecho crt.init).
    pub fn ensureDeviceBuffers(self: *Self, vocab: usize) !void {
        _ = self;
        _ = vocab;
        return error.NotImplemented;
    }

    fn addTrace(self: *Self, t: StepTrace) void {
        if (!debugz.dbg.dump_spec) return;
        if (self.traces.items.len >= self.trace_cap) self.traces.clearRetainingCapacity();
        self.traces.append(self.allocator, t) catch {};
    }

    /// Imprime las métricas de aceptación (llamar al fin de generación o con
    /// DUMP_SPEC tras cada ronda).
    pub fn reportMetrics(self: *Self, writer: anytype) !void {
        try writer.print("[spec] rondas={d} drafts={d} aceptados={d} ({d:.1}%) bonus={d} tok/ronda={d:.2} fuente drafter={d} lookup={d}\n", .{
            self.metrics.total_rounds,
            self.metrics.total_drafted,
            self.metrics.total_accepted,
            self.metrics.acceptanceRate() * 100.0,
            self.metrics.total_bonus,
            self.metrics.tokensPerRound(),
            self.metrics.source_drafter,
            self.metrics.source_lookup,
        });
        // 9.5: ProfitController. Solo imprime si hay datos adaptativos
        // (depth != 0 o decision != 0). Cuando el decode loop C4 enchufe
        // el controller, estos campos se actualizan por recordProfitDecision.
        if (self.metrics.adaptive_decision != 0 or self.metrics.adaptive_depth != 0) {
            try writer.print("[spec.adaptive] depth={d} decision={s} baseline_us={d:.1}\n", .{
                self.metrics.adaptive_depth,
                adaptiveDecisionName(self.metrics.adaptive_decision),
                self.metrics.adaptive_baseline_us,
            });
        }
    }

    /// Texto corto para la decisión adaptativa (9.5) — útil en logs.
    fn adaptiveDecisionName(tag: u8) []const u8 {
        return switch (tag) {
            0 => "pending",
            1 => "hold",
            2 => "demote",
            3 => "promote_step",
            4 => "promote_to_best",
            5 => "shutdown",
            else => "unknown",
        };
    }

    /// Verificación greedy de una ronda completa contra filas de logits del
    /// target ya materializadas en host (una fila por posición del bloque
    /// [contexto..., draft...]). Implementado CPU-side y testeable; el path
    /// device (D2H de filas desde logits_buf) lo enchufa C4.
    ///
    /// `out` debe tener capacidad >= drafts.len + 1.
    pub fn verifyGreedyRows(
        self: *Self,
        drafts: []const u32,
        target_rows: []const []const f32,
        out: []u32,
    ) AcceptResult {
        const n_before = self.metrics;
        _ = n_before;
        var filled: usize = 0;
        for (drafts, 0..) |dtok, pos| {
            if (pos >= target_rows.len) break;
            const ok = sampler.acceptGreedy(target_rows[pos], dtok);
            self.addTrace(.{ .pos = pos, .draft_token = dtok, .accepted = ok, .source = .drafter });
            if (!ok) break;
            out[filled] = dtok;
            filled += 1;
        }
        // Bonus: argmax del target en la última posición verificada.
        var n_total = filled;
        if (target_rows.len > 0 and out.len > n_total) {
            const row = target_rows[@min(filled, target_rows.len - 1)];
            out[n_total] = sampler.greedy(row);
            n_total += 1;
        }
        const res = AcceptResult{ .n_accepted = filled, .n_total = n_total };
        self.metrics.recordRound(drafts.len, res);
        if (debugz.dbg.dump_spec) {
            debugz.dbg.printLevel(.info, "[spec] ronda: drafts={d} aceptados={d} total={d}\n", .{ drafts.len, filled, n_total });
        }
        return res;
    }

    // ── Verificación SECUENCIAL (v1, estilo common_speculative) ────────────
    // El decode loop consume los drafts uno a uno con sus forwards normales:
    // cada forward produce el argmax del target `a`; el draft pendiente se
    // acepta iff a == draft. Sin batch no hay speedup aún, pero el camino
    // completo (cola, aceptación, rechazo, rollback de KV del draft,
    // métricas) queda ejercitado y la salida es greedy-idéntica por
    // construcción. C4.3 añadirá el forward batched del bloque.

    /// Estado de una ronda de especulación del decode loop.
    pub const Round = struct {
        /// Tokens draft pendientes de verificar (frente = siguiente posición).
        queue: std.ArrayList(u32),
        /// Longitud del KV del draft al empezar la ronda (para truncar).
        draft_base_len: usize = 0,
        /// Posición base del target al empezar la ronda.
        target_base_pos: usize = 0,

        pub fn init() Round {
            return .{ .queue = .empty };
        }

        pub fn deinit(self: *Round, allocator: std.mem.Allocator) void {
            self.queue.deinit(allocator);
            self.* = undefined;
        }

        pub fn reset(self: *Round) void {
            self.queue.clearRetainingCapacity();
        }

        pub fn pending(self: *const Round) usize {
            return self.queue.items.len;
        }
    };

    /// Registra el resultado de un paso de verificación secuencial.
    /// `accepted`: el argmax del target coincidió con el frente de la cola.
    /// Devuelve el token a commitir: el draft si acepta; `fallback` (el
    /// argmax del target) si rechaza o cola vacía.
    pub fn verifyStep(
        self: *Self,
        round: *Round,
        target_argmax: u32,
        fallback: u32,
        eos_reached: bool,
    ) u32 {
        if (round.queue.items.len == 0 or eos_reached) {
            // Fin de ronda natural: el token del target es el nuevo ancla.
            self.metrics.total_bonus += 1;
            return fallback;
        }
        const cand = round.queue.items[0];
        const ok = cand == target_argmax and !eos_reached;
        _ = round.queue.orderedRemove(0);
        self.addTrace(.{
            .pos = self.metrics.total_accepted + self.metrics.total_rejected_step,
            .draft_token = cand,
            .accepted = ok,
            .source = .drafter,
        });
        if (ok) {
            self.metrics.total_accepted += 1;
            self.metrics.source_drafter += 1;
            return cand;
        }
        // Rechazo: el argmax del target corrige la trayectoria; el resto de
        // la cola queda obsoleto (el caller trunca el KV del draft y resetea).
        self.metrics.total_rejected_step += 1;
        self.metrics.total_bonus += 1;
        return fallback;
    }

    /// Descarta el resto de la cola tras un rechazo (contabilidad).
    pub fn discardQueue(self: *Self, round: *Round) void {
        const dropped = round.pending();
        if (dropped > 0) {
            self.metrics.total_drafted += dropped;
            self.metrics.total_rejected_step += dropped;
            if (debugz.dbg.dump_spec) {
                debugz.dbg.printLevel(.info, "[spec] descartados {d} drafts tras rechazo\n", .{dropped});
            }
        }
        round.reset();
    }

    // ── Caminos device: stubs hasta C4 (MTP) / C6-C7 (sidecars) ────────────

    /// Draft MTP: forward de la cabeza nextn k veces alimentándose de
    /// (emb(tok), hidden). Requiere taps del target y KV de la capa nextn.
    pub fn draftMtp(self: *Self, last_token: u32, last_hidden: []const f32, out_tokens: []u32) SpecError!usize {
        _ = self;
        _ = last_token;
        _ = last_hidden;
        _ = out_tokens;
        return error.NotImplemented;
    }

    /// DFlash/DSpark/DFlash2: encoder fc+norm → KV-inject → denoise
    /// no-causal sobre [anchor, MASK×(bs-1)] → top-k con truncado p_min.
    /// 5.2 (lane-b1): el draft-model concreto (DflashDraftModel de
    /// dflash_draft.zig) vive en el caller (cli.zig — posee el KV paginado);
    /// este driver aporta el TRUNCADO p_min + métricas del source.
    pub fn draftSidecarGreedy(self: *Self, block_logits: []const f32, vocab: usize, out_tokens: []u32) SpecError!usize {
        // Greedy top-1 por posición con truncado p_min (softmax confidence,
        // mismo criterio que draftMtpLookup): primera posición con
        // p < p_min corta la cola.
        var n: usize = 0;
        for (0..out_tokens.len) |i| {
            const row = block_logits[i * vocab ..][0..vocab];
            const tok = sampler.greedy(row);
            const conf = sampler.softmaxConfidence(row, tok);
            if (i > 0 and conf < self.config.p_min) break;
            out_tokens[n] = tok;
            n += 1;
        }
        if (n > 0) self.metrics.source_drafter += 1;
        return n;
    }
};

test "detectMtp encuentra la capa nextn del GGUF NEO-MTP" {
    const io = std.Io.Threaded.global_single_threaded.io();
    // Modelo pequeño sin MTP: debe devolver null sin colgar.
    const path = std.c.getenv("GGUF_MODEL_PATH") orelse return error.SkipZigTest;
    var model = try gguf.GgufModel.load(io, std.heap.c_allocator, std.mem.span(path));
    defer model.deinit();
    const info = detectMtp(&model.file, model.config.block_count);
    if (info) |m| {
        try std.testing.expectEqual(model.config.block_count, m.layer_idx);
        try std.testing.expect(m.predict_layers >= 1);
    }
}

test "SpecType.fromCli cubre los valores de --spec-type" {
    try std.testing.expectEqual(SpecType.none, SpecType.fromCli("none").?);
    try std.testing.expectEqual(SpecType.draft_mtp, SpecType.fromCli("draft-mtp").?);
    try std.testing.expectEqual(SpecType.draft_dflash2, SpecType.fromCli("draft-dflash2").?);
    try std.testing.expect(SpecType.fromCli("no-existe") == null);
}

test "SpecDriver.verifyGreedyRows integra sampler y métricas" {
    const allocator = std.testing.allocator;
    var drv = SpecDriver.init(allocator, .{});
    defer drv.deinit();

    const row0 = [_]f32{ 0.0, 9.0, 1.0 };
    const row1 = [_]f32{ 1.0, 0.0, 9.0 };
    const rows = [_][]const f32{ &row0, &row1 };
    var out: [8]u32 = undefined;

    const res = drv.verifyGreedyRows(&.{ 1, 2 }, &rows, &out);
    try std.testing.expectEqual(@as(usize, 2), res.n_accepted);
    try std.testing.expectEqual(@as(usize, 3), res.n_total);
    try std.testing.expectEqual(@as(u64, 1), drv.metrics.total_rounds);
    try std.testing.expectEqual(@as(f32, 1.0), drv.metrics.acceptanceRate());
}
