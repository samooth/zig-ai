//! KLD measurement tool (lane-b3, P3.5).
//!
//! Mide KL-divergence del pipeline KV (candidato cuantizado) contra un
//! baseline BF16:
//!
//!   zig-ai-engine kld -m model.gguf --kld-corpus corpus.txt \
//!       -ctk kvarn5 -ctv kvarn4 --kv-tail-tokens 1024 \
//!       -b 2048 -ub 512 --seed 42
//!
//! ## Reglas Bee de benchmarks (obligatorias)
//! - Mismo `-b` y `-ub` en baseline y candidato (se valida al parsear:
//!   la corrida compara SOLO el formato KV).
//! - Output reproducible: commit + comando + corpus + seed + GPU + sampling.
//! - Formato tabla mediana + p99.9 KLD (mismo formato que ladder Bee README).
//!
//! ## Diseño
//! El núcleo (`kldFromLogits`) es puro: dos buffers de logits → escalar.
//! La orchestración (cargar modelo, prefill del corpus, capturar logits)
//! vive en `run()`, que el wiring P4 conecta como subcomando `kld` de
//! main.zig. Los tests de sanity corren 100% CPU sobre el núcleo puro.
const std = @import("std");
pub const debugz = @import("debug");

/// Config del tool (subcomando `kld`).
pub const KldConfig = struct {
    /// Modelo GGUF (-m).
    model_path: []const u8 = "",
    /// Corpus de texto (--kld-corpus).
    corpus_path: []const u8 = "",
    /// Formato K del candidato (-ctk). El baseline SIEMPRE es bf16.
    cache_type_k: []const u8 = "bf16",
    /// Formato V del candidato (-ctv).
    cache_type_v: []const u8 = "bf16",
    /// Cola exacta f16 (--kv-tail-tokens; 0 = sin cola).
    kv_tail_tokens: usize = 0,
    /// Batch de prefill (-b). Debe ser IGUAL en baseline y candidato.
    batch_size: usize = 2048,
    /// Micro-batch (-ub). Debe ser IGUAL en baseline y candidato.
    ubatch_size: usize = 512,
    /// Seed determinista (--seed).
    seed: u64 = 42,
    /// Percentiles del report.
    report_percentiles: bool = true,

    /// Bee rule: -b y -ub emparejados. El tool NO permite comparar
    /// corridas con batches distintos — la validación es parte del parse.
    pub fn validateBatches(self: KldConfig) !void {
        if (self.batch_size == 0 or self.ubatch_size == 0) {
            return error.InvalidBatchConfig;
        }
        if (self.ubatch_size > self.batch_size) {
            return error.UbatchGreaterThanBatch;
        }
    }
};

/// Resultado de una corrida KLD sobre N posiciones.
pub const KldResult = struct {
    /// KLD por posición (nats).
    samples: []const f64,
    /// Mediana de samples.
    pub fn median(self: KldResult) f64 {
        if (self.samples.len == 0) return 0;
        var buf: [4096]f64 = undefined;
        const n = @min(self.samples.len, buf.len);
        @memcpy(buf[0..n], self.samples[0..n]);
        std.mem.sort(f64, buf[0..n], {}, std.sort.asc(f64));
        if (n % 2 == 1) return buf[n / 2];
        return (buf[n / 2 - 1] + buf[n / 2]) / 2.0;
    }

    /// Percentil p (0..100) por interpolación de rango más cercano.
    pub fn percentile(self: KldResult, p: f64) f64 {
        if (self.samples.len == 0) return 0;
        var buf: [4096]f64 = undefined;
        const n = @min(self.samples.len, buf.len);
        @memcpy(buf[0..n], self.samples[0..n]);
        std.mem.sort(f64, buf[0..n], {}, std.sort.asc(f64));
        if (n == 1) return buf[0];
        const idx_f = (p / 100.0) * @as(f64, @floatFromInt(n - 1));
        const lo: usize = @intFromFloat(@floor(idx_f));
        const hi: usize = @intFromFloat(@ceil(idx_f));
        if (lo == hi) return buf[lo];
        const frac = idx_f - @floor(idx_f);
        return buf[lo] * (1.0 - frac) + buf[hi] * frac;
    }

    /// Formato ladder Bee (mediana + p99.9 en tabla fija).
    pub fn formatLadder(self: KldResult, writer: anytype, label: []const u8) !void {
        try writer.print("| {s: <28} | median={d:.6} | p99.9={d:.6} | n={d} |\n", .{
            label,
            self.median(),
            self.percentile(99.9),
            self.samples.len,
        });
    }
};

/// KL-divergence por posición: KL(P‖Q) donde P = baseline (bf16) y
/// Q = candidato (KV cuantizado). Logits sin normalizar → softmax
/// numéricamente estable con máximos por fila.
///
/// KL(P‖Q) = Σ_i P_i · ln(P_i / Q_i)
///
/// Con P == Q (mismo buffer bitwise) el resultado es EXACTAMENTE 0
/// (propiedad que el sanity test verifica bit a bit).
pub fn kldFromLogits(p_logits: []const f32, q_logits: []const f32) f64 {
    std.debug.assert(p_logits.len == q_logits.len);
    if (p_logits.len == 0) return 0;

    // Softmax estable: restar el máximo de CADA distribución.
    var p_max: f32 = -std.math.floatMax(f32);
    var q_max: f32 = -std.math.floatMax(f32);
    for (p_logits, q_logits) |p, q| {
        if (p > p_max) p_max = p;
        if (q > q_max) q_max = q;
    }
    var p_sum: f64 = 0;
    var q_sum: f64 = 0;
    for (p_logits, q_logits) |p, q| {
        p_sum += @exp(@as(f64, p) - p_max);
        q_sum += @exp(@as(f64, q) - q_max);
    }

    var kl: f64 = 0;
    for (p_logits, q_logits) |p, q| {
        const p_prob = @exp(@as(f64, p) - p_max) / p_sum;
        const q_prob = @exp(@as(f64, q) - q_max) / q_sum;
        if (p_prob > 0) {
            // P_i * ln(P_i / Q_i); con P==Q bitwise, ln(1)=0 exacto.
            kl += p_prob * std.math.log(f64, std.math.e, p_prob / q_prob);
        }
    }
    return kl;
}

/// Conveniencia: KLD sobre una lista de posiciones → KldResult.
/// El caller retiene ownership de `samples` (el result lo copia por
/// referencia — NO liberar antes de terminar el report).
pub fn kldOverPositions(
    allocator: std.mem.Allocator,
    baseline: []const []const f32,
    candidate: []const []const f32,
) !KldResult {
    std.debug.assert(baseline.len == candidate.len);
    const samples = try allocator.alloc(f64, baseline.len);
    errdefer allocator.free(samples);
    for (baseline, candidate, 0..) |p, q, i| {
        samples[i] = kldFromLogits(p, q);
    }
    return .{ .samples = samples };
}

/// Report completo de una corrida: header reproducible + tabla ladder.
/// `meta` = commit + comando + GPU + sampling (regla Bee).
pub fn writeReport(writer: anytype, cfg: KldConfig, result: KldResult) !void {
    try writer.print("# KLD report — reproducibilidad Bee\n", .{});
    try writer.print("#   model: {s}\n", .{cfg.model_path});
    try writer.print("#   corpus: {s}\n", .{cfg.corpus_path});
    try writer.print("#   kv: k={s} v={s} tail={d}\n", .{ cfg.cache_type_k, cfg.cache_type_v, cfg.kv_tail_tokens });
    try writer.print("#   batch: -b {d} -ub {d} (emparejados baseline/candidato)\n", .{ cfg.batch_size, cfg.ubatch_size });
    try writer.print("#   seed: {d}\n", .{cfg.seed});
    try writer.print("\n", .{});
    try result.formatLadder(writer, "kld");
}

/// Punto de entrada del subcomando `kld` (wiring P4 en main.zig).
/// El pipeline real (carga GGUF + prefill corpus + captura logits) se
/// conecta cuando B2 cierre P0.1 (KVarN real); la huella de la interfaz
/// queda congelada aquí.
pub fn run(allocator: std.mem.Allocator, cfg: KldConfig) !void {
    try cfg.validateBatches();
    _ = allocator;
    // TODO(lane-b3, post-B2-P0.1): orchestración del pipeline.
    //   1. Cargar modelo bf16 (baseline) y candidato (-ctk/-ctv).
    //   2. Prefill del corpus por chunks de -b con ubatch -ub (IGUAL en ambos).
    //   3. Capturar logits por posición en ambos.
    //   4. kldOverPositions + writeReport a stdout.
    return error.NotImplemented;
}

// ─── Tests sanity (100% CPU, sin GPU, sin modelo) ────────────────────────────

test "KLD sanity: bf16 vs bf16 = 0 exacto" {
    const logits = [_]f32{ 0.1, -2.0, 0.5, 5.0, -0.3, 1.7 };
    const kl = kldFromLogits(&logits, &logits);
    try std.testing.expectEqual(@as(f64, 0), kl);
}

test "KLD sanity: distribución idéntica con logits desplazados = 0" {
    // Softmax es invariante a shifts: P y Q iguales modulo constante ⇒ KL=0
    // (aunque los logits raw difieran).
    const base = [_]f32{ 0.1, -2.0, 0.5, 5.0 };
    var shifted: [4]f32 = undefined;
    for (&shifted, base) |*s, b| s.* = b + 10.0;
    const kl = kldFromLogits(&base, &shifted);
    try std.testing.expectApproxEqAbs(@as(f64, 0), kl, 1e-12);
}

test "KLD sanity: perturbación pequeña → KLD pequeño y positivo" {
    const base = [_]f32{ 0.1, -2.0, 0.5, 5.0 };
    var pert: [4]f32 = undefined;
    for (&pert, base) |*s, b| s.* = b + 0.01;
    const kl = kldFromLogits(&base, &pert);
    try std.testing.expect(kl > 0);
    try std.testing.expect(kl < 0.01);
}

test "KLD: mayor divergencia → mayor KLD (monotonía)" {
    // Perturbaciones NO constantes (softmax es invariante a shifts):
    // small[i] alterna signo, large[i] escala el doble.
    const base = [_]f32{ 0.0, 1.0, 2.0, 3.0 };
    var small: [4]f32 = undefined;
    var large: [4]f32 = undefined;
    for (&small, &large, base, 0..) |*s, *l, b, i| {
        const sign: f32 = if (i % 2 == 0) 1.0 else -1.0;
        s.* = b + sign * 0.05;
        l.* = b + sign * 1.0;
    }
    const kl_small = kldFromLogits(&base, &small);
    const kl_large = kldFromLogits(&base, &large);
    try std.testing.expect(kl_small > 0);
    try std.testing.expect(kl_large > kl_small);
}

test "KLD: determinista bajo seed fija (3 corridas idénticas)" {
    var logits_a: [64]f32 = undefined;
    var logits_b: [64]f32 = undefined;
    const kls: [3]f64 = blk: {
        var acc: [3]f64 = undefined;
        for (0..3) |iter| {
            // Misma seed ⇒ mismo par de distribuciones ⇒ mismo KLD.
            var prng = std.Random.DefaultPrng.init(42);
            for (&logits_a, &logits_b) |*a, *b| {
                a.* = prng.random().float(f32) * 4.0 - 2.0;
                b.* = a.* + 0.02;
            }
            acc[iter] = kldFromLogits(&logits_a, &logits_b);
        }
        break :blk acc;
    };
    try std.testing.expectEqual(kls[0], kls[1]);
    try std.testing.expectEqual(kls[1], kls[2]);
}

test "KldResult: mediana y percentiles sobre samples ordenables" {
    const samples = [_]f64{ 0.1, 0.3, 0.2, 0.5, 0.4, 1.0, 0.9 };
    const r = KldResult{ .samples = &samples };
    // Mediana de 7 samples = 0.4 (ordenado: 0.1 0.2 0.3 0.4 0.5 0.9 1.0).
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), r.median(), 1e-12);
    // p99.9 interpola entre los dos mayores.
    const p999 = r.percentile(99.9);
    try std.testing.expect(p999 >= 0.9);
    try std.testing.expect(p999 <= 1.0);
}

test "KldConfig.validateBatches: -ub > -b rechazado" {
    var cfg = KldConfig{ .batch_size = 512, .ubatch_size = 2048 };
    try std.testing.expectError(error.UbatchGreaterThanBatch, cfg.validateBatches());
    cfg = .{ .batch_size = 0, .ubatch_size = 0 };
    try std.testing.expectError(error.InvalidBatchConfig, cfg.validateBatches());
    cfg = .{ .batch_size = 2048, .ubatch_size = 512 };
    try cfg.validateBatches();
}

test "kldOverPositions: consistente con kldFromLogits" {
    const row0_a = [_]f32{ 1.0, 2.0, 3.0 };
    const row1_a = [_]f32{ 0.5, 0.5, 0.5 };
    const row0_b = [_]f32{ 1.1, 2.1, 3.1 };
    const row1_b = [_]f32{ 0.5, 0.5, 0.5 };
    const baseline = [_][]const f32{ &row0_a, &row1_a };
    const candidate = [_][]const f32{ &row0_b, &row1_b };
    const r = try kldOverPositions(std.testing.allocator, &baseline, &candidate);
    defer std.testing.allocator.free(r.samples);
    try std.testing.expectEqual(@as(usize, 2), r.samples.len);
    try std.testing.expectEqual(kldFromLogits(&row0_a, &row0_b), r.samples[0]);
    // Fila idéntica → 0 exacto.
    try std.testing.expectEqual(@as(f64, 0), r.samples[1]);
}
