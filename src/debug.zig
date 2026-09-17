//! Instrumentación de diagnóstico centralizada.
//!
//! Toda la instrumentación vive en este módulo como "breadcrumbs": el código de
//! captura/dump está SIEMPRE presente en el árbol (no se borra), pero cada
//! punto solo emite si su flag está activo. Los flags se leen UNA vez de env en
//! `init()`; después son consultas baratas a campos del struct.
//!
//! Niveles (`DEBUG_LEVEL`, jerárquicos):
//!   0 = off      (por defecto: sin salida de diagnóstico)
//!   1 = info     (resumen de eventos relevantes)
//!   2 = detail   (datos por token/capa)
//!   3 = trace    (todo, valores completos)
//!
//! Flags por-dump (activan su breadcrumb independientemente del nivel):
//!   DUMPKV, DUMPNORM, DUMP_LOGITS, CHKSTATE, PERF_STAGE, DUMP_GRAPH,
//!   DUMP_KVQUANT (roundtrip encode/dequant KV cuantizado, escalas, stats
//!   pre/post append; activa también el diagnóstico interno del kernel
//!   kvAppendQ8_0Kernel vía su parámetro `dbg`),
//!   DUMP_SPEC (drafts generados, aceptación/rechazo por paso, fuente
//!   drafter-vs-lookup, confianzas del driver especulativo),
//!   PERF_SPEC (timing por etapa draft/verify/accept del driver especulativo)
//! Flags lane-b2 (BeeLlama):
//!   DUMP_KVARN (roundtrip encode/dequant KVarN, scales, stats pre/post kernel append),
//!   DUMP_KV_TAIL (estado cola exacta vs cuerpo comprimido, rollback KVCPT)
//! Flags RLT (Recurrent Looped Transformer):
//!   DUMP_RLT_STATE (stats de α y estado recurrente por token),
//!   PERF_RLT (timing del merge feedback por token)
//!
//! Flags de comportamiento (también centralizados):
//!   NOGRAPH, NOGPU_PREFILL, NOQ4, NOQ4ATTN, NOQ4SSM, NOQ4FFN, NOSSM4DP4A
//!
//! Scope filter (`DEBUG_SCOPE`): solo emite breadcrumbs con `[tag]` que
//! coincida con uno de los tags separados por comas. Sin DEBUG_SCOPE = todo pasa.
//! Coste: cero cuando no hay scope activo (`scope_count == 0`); cuando hay
//! scope activo, comparación runtime contra hasta 8 entradas.
//!   DEBUG_SCOPE=kv,ssm,paged_attn
//!
//! Layer filter (`DEBUG_LAYER=N`): solo emite breadcrumbs cuyo primer arg
//! entero coincida con N. Sin DEBUG_LAYER = todo pasa.
//!   DEBUG_LAYER=5
//!
//! Structured output (`DEBUG_FORMAT=json`): emite JSON line por cada
//! printLevel, útil para diff automatizado entre runs.
const builtin = @import("builtin");
const std = @import("std");

pub const Level = enum(u8) {
    off = 0,
    info = 1,
    detail = 2,
    trace = 3,

    pub fn fromEnv() Level {
        return parseLevel(std.c.getenv("DEBUG_LEVEL"));
    }
};

fn parseLevel(s: ?[*:0]const u8) Level {
    if (s) |v| {
        if (std.fmt.parseInt(u8, std.mem.span(v), 10)) |n| {
            if (n >= @intFromEnum(Level.off) and n <= @intFromEnum(Level.trace)) {
                return @enumFromInt(n);
            }
        } else |_| {}
    }
    return .off;
}

pub const Debug = struct {
    level: Level = .off,

    // Breadcrumbs de dump (env: DUMPKV, DUMPNORM, ...).
    dump_kv: bool = false,
    dump_norm: bool = false,
    dump_logits: bool = false,
    chk_state: bool = false,
    perf_stage: bool = false,
    dump_graph: bool = false,
    dump_prefill_layers: bool = false,
    dump_kvquant: bool = false,
    /// Valor crudo de DUMP_KVQUANT (2 = modo diagnóstico append sin tiendas:
    /// el kernel lee fuentes pero no escribe al pool, para aislar lecturas
    /// vs escrituras en fallos asíncronos).
    dump_kvquant_val: u8 = 0,
    /// Driver especulativo: drafts, aceptación/rechazo por paso, fuente
    /// (drafter vs lookup), confianzas (env: DUMP_SPEC).
    dump_spec: bool = false,
    /// Driver especulativo: timing por etapa draft/verify/accept
    /// (env: PERF_SPEC).
    perf_spec: bool = false,
    /// Lane-B2: roundtrip encode/dequant KVarN, scales Sinkhorn, stats
    /// pre/post append (env: DUMP_KVARN).
    dump_kvarn: bool = false,
    /// Lane-B2: estado cola exacta vs cuerpo comprimido, rollback KVCPT
    /// (env: DUMP_KV_TAIL).
    dump_kv_tail: bool = false,
    // Vision (PLAN_MMPROJ, lane-mmproj): dumps por etapa del encoder ViT
    // (conv/inp/pe/ln1/qkv/rope/aout/res1/lout) y timing del encode.
    dump_mm_input: bool = false,
    perf_mm: bool = false,
    /// Lane-B1 (Dev-B): contadores por ruta FA KVarN (portable/split/vec/
    /// generic/fallback) + geometry elegida (env: PERF_KVARN_ROUTE).
    perf_kvarn_route: bool = false,
    /// RLT: stats de α y estado recurrente por token (env: DUMP_RLT_STATE).
    dump_rlt_state: bool = false,
    /// RLT: timing del merge feedback por token (env: PERF_RLT).
    perf_rlt: bool = false,
    /// Lane-KVC (paso 1 KV-Codec): ruta de captura de trazas K/V del forward
    /// CPU (env: DUMP_KV_TRACE=<dir>). null = off, coste cero absoluto.
    kv_trace_dir: ?[]const u8 = null,
    /// RLT train (fase 2): directorio de captura de hidden states para el
    /// entrenamiento del adapter (env: RLT_CAPTURE_DIR=<dir>). null = off.
    /// Requiere prefill CPU (NOGPU_PREFILL=1 recomendado para dets).
    rlt_capture_dir: ?[]const u8 = null,
    /// RLT train: lista de capas a capturar (env: RLT_CAPTURE_LAYERS=0,4,8).
    /// null = todas.
    rlt_capture_layers: ?[]const u8 = null,

    // Flags de comportamiento (env: NOGRAPH, NOQ4, NOFP8...).
    no_graph: bool = false,
    no_gpu_prefill: bool = false,
    no_q4: bool = false,
    no_q4_attn: bool = false,
    no_q4_ssm: bool = false,
    no_q4_ffn: bool = false,
    /// G0 (TODO 1.9, lane-c): familia SSM4DP4A (GEMV dp4a q4_0/q5_k/q6_k
    /// M=1 y M≤32) DEFAULT-ON — medido 75.9→150.4 t/s (+98%), paridad
    /// e2e byte-idéntica @ee75066. NOSSM4DP4A=1 la desactiva (A/B).
    no_ssm4dp4a: bool = false,
    /// KVSR_V=1: stochastic rounding en el encoder V (paso 0 KV-Codec
    /// §4.6 — sesgo neto 4× menor que redondeo determinístico, medido).
    kv_sr_v: bool = false,
    /// G2 (TODO 1.7, lane-a): argmax en device + fin de grafo en device.
    ///   Hoy el decode copia TODO el vector de logits a host (vocab·4B ≈
    ///   993 KB ≈ 257 µs) y hace el argmax en CPU (~84 µs); con esto el
    ///   argmax corre en GPU (`argmaxF32Kernel`) y el D2H es de 4 bytes.
    ///   DEFAULT-ON desde 2026-09-08 (medido Qwen3.5-0.8B-Q4_0, greedy,
    ///   -n 64 ×3 interleaved: 175.25 → 184.23 tok/s = **+5.1%**, salida
    ///   BYTE-IDÉNTICA en las 6 corridas — hash md5 del texto generado
    ///   idéntico con y sin el atajo). `NOGPUARGMAX=1` lo desactiva.
    /// Sólo aplica a greedy puro (temp<=0 y sin repetition penalty) y con
    /// los logits producidos en device — el resto de muestreadores
    /// necesita los logits completos en host y sigue el camino original.
    gpu_argmax: bool = true,
    // FP8 flags
    no_fp8: bool = false,
    no_fp8_attn: bool = false,
    no_fp8_ffn: bool = false,
    /// F-4 (lane-f): A/B del residual stream legacy — ON=f16 clásico
    /// (pre-fix 1.8× PPL), unset=f32 (default post @53a37ab).
    no_f32_stream: bool = false,
    /// RLT (Recurrent Looped Transformer): gated recurrent feedback merge.
    ///   null = auto (ON if GGUF has rlt.feedback_alpha > 0 + weights),
    ///   true = force ON (even without GGUF weights, uses alpha=0.1 default),
    ///   false = force OFF (skip merge entirely, zero overhead).
    ///   Env: ZIG_AI_RLT_FEEDBACK=1 force ON, =0 force OFF, unset = auto.
    rlt_feedback: ?bool = null,

    // Scope filter (env: DEBUG_SCOPE=kv,ssm,paged_attn).
    scope_set: [8][16]u8 = undefined,
    scope_count: u8 = 0,
    // Layer filter (env: DEBUG_LAYER=N): solo breadcrumbs cuyo primer arg
    // entero coincida con N. Sin DEBUG_LAYER = todo pasa.
    layer: ?u32 = null,
    // Structured output (env: DEBUG_FORMAT=json).
    json_output: bool = false,
    /// Dump estructurado adicional a archivo JSONL (env: DEBUG_JSON_DUMP=/path).
    json_dump: ?[:0]const u8 = null,

    pub fn init() Debug {
        var d = Debug{
            .level = Level.fromEnv(),
            .dump_kv = envOn("DUMPKV"),
            .dump_norm = envOn("DUMPNORM"),
            .dump_logits = envOn("DUMP_LOGITS"),
            .chk_state = envOn("CHKSTATE"),
            .perf_stage = envOn("PERF_STAGE"),
            .dump_graph = envOn("DUMP_GRAPH"),
            .dump_prefill_layers = envOn("DUMP_PREFILL_LAYERS"),
            .dump_kvquant = envOn("DUMP_KVQUANT"),
            .dump_kvquant_val = blk: {
                const v = std.c.getenv("DUMP_KVQUANT");
                if (v) |p| {
                    if (std.fmt.parseInt(u8, std.mem.span(p), 10)) |n| break :blk n else |_| {}
                }
                break :blk if (envOn("DUMP_KVQUANT")) 1 else 0;
            },
            .dump_spec = envOn("DUMP_SPEC"),
            .perf_spec = envOn("PERF_SPEC"),
            .dump_kvarn = envOn("DUMP_KVARN"),
            .dump_kv_tail = envOn("DUMP_KV_TAIL"),
            .dump_mm_input = envOn("DUMP_MM_INPUT"),
            .perf_mm = envOn("PERF_MM"),
            .perf_kvarn_route = envOn("PERF_KVARN_ROUTE"),
            .dump_rlt_state = envOn("DUMP_RLT_STATE"),
            .perf_rlt = envOn("PERF_RLT"),
            .kv_trace_dir = blk: {
                const v = std.c.getenv("DUMP_KV_TRACE");
                if (v) |p| {
                    const s = std.mem.span(p);
                    if (s.len > 0) break :blk s;
                }
                break :blk null;
            },
            .rlt_capture_dir = blk: {
                const v = std.c.getenv("RLT_CAPTURE_DIR");
                if (v) |p| {
                    const s = std.mem.span(p);
                    if (s.len > 0) break :blk s;
                }
                break :blk null;
            },
            .rlt_capture_layers = blk: {
                const v = std.c.getenv("RLT_CAPTURE_LAYERS");
                if (v) |p| {
                    const s = std.mem.span(p);
                    if (s.len > 0) break :blk s;
                }
                break :blk null;
            },
            .no_graph = envOn("NOGRAPH"),
            .no_gpu_prefill = envOn("NOGPU_PREFILL"),
            .no_q4 = envOn("NOQ4"),
            .no_q4_attn = envOn("NOQ4ATTN"),
            .no_q4_ssm = envOn("NOQ4SSM"),
            .no_q4_ffn = envOn("NOQ4FFN"),
            .kv_sr_v = envOn("KVSR_V"),
            .no_ssm4dp4a = envOn("NOSSM4DP4A"),
            .gpu_argmax = !envOn("NOGPUARGMAX"),
            .no_fp8 = envOn("NOFP8"),
            .no_fp8_attn = envOn("NOFP8ATTN"),
            .no_fp8_ffn = envOn("NOFP8FFN"),
            .no_f32_stream = envOn("NOF32STREAM"),
            .rlt_feedback = blk_rlt: {
                const v = std.c.getenv("ZIG_AI_RLT_FEEDBACK");
                if (v) |p| {
                    const s = std.mem.span(p);
                    if (s.len == 0) break :blk_rlt null;
                    if (std.mem.eql(u8, s, "0")) break :blk_rlt false;
                    if (std.mem.eql(u8, s, "1")) break :blk_rlt true;
                }
                break :blk_rlt null; // auto
            },
            // FIX (lane-b1): DEBUG_FORMAT solo activa con =json exacto (antes
            // DEBUG_FORMAT=0/false activaba igual — getenv != null).
            .json_output = blk_fmt: {
                const v = std.c.getenv("DEBUG_FORMAT");
                break :blk_fmt v != null and std.mem.eql(u8, std.mem.span(v.?), "json");
            },
            .json_dump = blk_dump: {
                const v = std.c.getenv("DEBUG_JSON_DUMP");
                break :blk_dump if (v) |p| blk: {
                    const s = std.mem.span(p);
                    if (s.len == 0) break :blk null;
                    break :blk std.mem.span(p);
                } else null;
            },
        };
        // Parse DEBUG_SCOPE=tag1,tag2,...
        if (std.c.getenv("DEBUG_SCOPE")) |scope_str| {
            var it = std.mem.splitScalar(u8, std.mem.span(scope_str), ',');
            var count: u8 = 0;
            while (it.next()) |token| {
                if (count >= 8) break;
                const t = std.mem.trim(u8, token, " ");
                if (t.len == 0 or t.len > 15) continue;
                // FIX (ambos lanes, misma semántica): @memcpy exige longitudes
                // iguales — copiar el slice EXACTO del sub-rango [0..t.len].
                @memcpy(d.scope_set[count][0..t.len], t[0..t.len]);
                d.scope_set[count][t.len] = 0;
                count += 1;
            }
            d.scope_count = count;
        }
        // Parse DEBUG_LAYER=N
        if (std.c.getenv("DEBUG_LAYER")) |layer_str| {
            if (std.fmt.parseInt(u32, std.mem.span(layer_str), 10)) |n| {
                d.layer = n;
            } else |_| {}
        }

        return d;
    }

    pub fn at(self: Debug, lvl: Level) bool {
        return @intFromEnum(self.level) >= @intFromEnum(lvl);
    }

    pub fn print(self: Debug, comptime fmt: []const u8, args: anytype) void {
        if (self.level == .off) return;
        // FIX: print() también respeta el scope filter (inconsistencia con
        // printLevel — sin esto, DEBUG_SCOPE no filtraba nada emitido con print).
        if (!self.matchesScope(fmt)) return;
        if (self.json_output) {
            printJson(.info, fmt, args);
        } else {
            std.debug.print(fmt, args);
        }
    }

    pub fn printLevel(self: Debug, lvl: Level, comptime fmt: []const u8, args: anytype) void {
        if (!self.at(lvl)) return;
        if (!self.matchesScope(fmt)) return;
        if (!self.matchesLayer(args)) return;
        if (self.json_output) {
            printJson(lvl, fmt, args);
        } else {
            std.debug.print(fmt, args);
        }
    }

    /// Layer filter: si DEBUG_LAYER=N, solo pasa si el primer arg entero == N.
    fn matchesLayer(self: Debug, args: anytype) bool {
        if (self.layer == null or args.len == 0) return true;
        const Arg = @TypeOf(args[0]);
        const info = @typeInfo(Arg);
        if (info == .int) {
            return @as(u32, @intCast(args[0])) == self.layer.?;
        }
        return true;
    }

    fn printJson(lvl: Level, comptime fmt: []const u8, args: anytype) void {
        const ts = monotonicNs() / std.time.ns_per_ms;
        const tag = comptime blk: {
            if (fmt.len < 2 or fmt[0] != '[') break :blk "";
            var i: usize = 1;
            while (i < fmt.len and fmt[i] != ']') i += 1;
            if (i >= fmt.len) break :blk "";
            break :blk fmt[1..i];
        };
        const prefix_has_specs = comptime blk_p: {
            if (tag.len == 0) break :blk_p false;
            break :blk_p std.mem.indexOf(u8, fmt[0 .. 1 + tag.len + 1], "{") != null;
        };
        const body = comptime blk_b: {
            if (tag.len == 0 or prefix_has_specs) break :blk_b fmt;
            var j = 1 + tag.len + 1;
            if (j < fmt.len and fmt[j] == ' ') j += 1;
            break :blk_b fmt[j..];
        };
        const tag_out = if (prefix_has_specs) "" else tag;
        const lvl_str = switch (lvl) {
            .off => "off",
            .info => "info",
            .detail => "detail",
            .trace => "trace",
        };
        var buf: [4096]u8 = undefined;
        var fb = std.fmt.bufPrint(&buf, "{{\"ts\":{d},\"level\":\"{s}\",\"tag\":\"{s}\",\"msg\":\"", .{ ts, lvl_str, tag_out }) catch blk: {
            const max_fmt: usize = @min(fmt.len, buf.len / 4);
            break :blk std.fmt.bufPrint(&buf, "{s}...[TRUNC]", .{fmt[0..max_fmt]}) catch return;
        };
        var end = fb.len;
        while (end > 0 and (fb[end - 1] == '\n' or fb[end - 1] == '\r')) end -= 1;
        const prefix = fb[0..end];
        var msg_buf: [4096]u8 = undefined;
        const msg = std.fmt.bufPrint(&msg_buf, body, args) catch blk: {
            const max_fmt: usize = @min(body.len, msg_buf.len / 4);
            break :blk std.fmt.bufPrint(&msg_buf, "{s}...[TRUNC]", .{body[0..max_fmt]}) catch return;
        };
        var m_end = msg.len;
        while (m_end > 0 and (msg[m_end - 1] == '\n' or msg[m_end - 1] == '\r')) m_end -= 1;
        const trimmed = msg[0..m_end];
        var out_buf: [8192]u8 = undefined;
        var out_len: usize = 0;
        const dst_prefix = out_buf[0..prefix.len];
        @memcpy(dst_prefix, prefix);
        out_len += prefix.len;
        printEscapedToBuf(trimmed, out_buf[0..], &out_len);
        const suffix = "\"}\n";
        const dst_suffix = out_buf[out_len..][0..suffix.len];
        @memcpy(dst_suffix, suffix);
        out_len += suffix.len;
        std.debug.print("{s}", .{out_buf[0..out_len]});
        if (dbg.json_dump) |dump_path| {
            dumpJsonLine(dump_path, out_buf[0..out_len]);
        }
    }

    fn dumpJsonLine(path: [:0]const u8, line: []const u8) void {
        const dir = std.Io.Dir.cwd();
        const file = dir.createFile(std.Io.Threaded.global_single_threaded.io(), path, .{ .truncate = false }) catch return;
        defer file.close(std.Io.Threaded.global_single_threaded.io());
        var buf: [4096]u8 = undefined;
        var w = file.writer(std.Io.Threaded.global_single_threaded.io(), &buf);
        w.interface.writeAll(line) catch {};
        w.interface.flush() catch {};
    }

    /// Escribe `s` escapado en `out_buf` avanzando `out_len`.
    fn printEscapedToBuf(s: []const u8, out_buf: []u8, out_len: *usize) void {
        var i: usize = 0;
        while (i < s.len) : (i += 1) {
            const c = s[i];
            if (out_len.* + 4 >= out_buf.len) break;
            const dst = out_buf[out_len.*..][0..4];
            switch (c) {
                '"' => {
                    @memcpy(dst[0..2], "\\\"");
                    out_len.* += 2;
                },
                '\\' => {
                    @memcpy(dst[0..2], "\\\\");
                    out_len.* += 2;
                },
                '\n' => {
                    @memcpy(dst[0..2], "\\n");
                    out_len.* += 2;
                },
                '\r' => {
                    @memcpy(dst[0..2], "\\r");
                    out_len.* += 2;
                },
                '\t' => {
                    @memcpy(dst[0..2], "\\t");
                    out_len.* += 2;
                },
                else => {
                    if (c < 0x20) {
                        const esc = std.fmt.bufPrint(dst, "\\u{x:0>4}", .{c}) catch break;
                        out_len.* += esc.len;
                    } else {
                        dst[0] = c;
                        out_len.* += 1;
                    }
                },
            }
        }
    }

    /// Como printEscaped pero trimea whitespace final del formateado.
    fn printEscapedTrimmed(comptime fmt: []const u8, args: anytype) void {
        var buf: [4096]u8 = undefined;
        const s = std.fmt.bufPrint(&buf, fmt, args) catch blk: {
            const max_fmt: usize = @min(fmt.len, buf.len / 4);
            break :blk std.fmt.bufPrint(&buf, "{s}...[TRUNC]", .{fmt[0..max_fmt]}) catch return;
        };
        var end = s.len;
        while (end > 0 and (s[end - 1] == '\n' or s[end - 1] == '\r')) end -= 1;
        printEscapedStr(s[0..end]);
    }

    /// Emite `fmt,args` con escapes JSON sobre el RESULTADO formateado.
    /// (Los escapes deben aplicarse post-formato: los caracteres especiales
    /// pueden venir tanto del fmt como de los args.)
    fn printEscaped(comptime fmt: []const u8, args: anytype) void {
        var buf: [4096]u8 = undefined;
        const s = std.fmt.bufPrint(&buf, fmt, args) catch {
            // Mensaje más largo que el buffer: el fmt comptime cabe
            // (slices de args demasiado largos) — truncar el fmt a
            // buf-len de escape y reintentar.
            const max_fmt: usize = @min(fmt.len, buf.len / 4);
            const s2 = std.fmt.bufPrint(&buf, "{s}...[TRUNC]", .{fmt[0..max_fmt]}) catch return;
            printEscapedStr(s2);
            return;
        };
        printEscapedStr(s);
    }

    fn printEscapedStr(s: []const u8) void {
        for (s) |c| {
            switch (c) {
                '"' => std.debug.print("\\\"", .{}),
                '\\' => std.debug.print("\\\\", .{}),
                '\n' => std.debug.print("\\n", .{}),
                '\r' => std.debug.print("\\r", .{}),
                '\t' => std.debug.print("\\t", .{}),
                // Control chars no printable → \uXXXX (JSON exige escape).
                else => {
                    if (c < 0x20) {
                        std.debug.print("\\u{x:0>4}", .{c});
                    } else {
                        std.debug.print("{c}", .{c});
                    }
                },
            }
        }
    }

    /// Scope filter público: verifica si un `tag` ya extraído está dentro del
    /// scope_set activo. Sin scope activo (`scope_count == 0`) → siempre pasa.
    /// Coste: cero cuando no hay scope activo.
    pub fn scopeMatches(self: Debug, tag: []const u8) bool {
        if (tag.len == 0) return true;
        if (self.scope_count == 0) return true;
        for (0..8) |i| {
            if (i >= self.scope_count) return false;
            const scope_ptr: [*:0]const u8 = @ptrCast(&self.scope_set[i]);
            if (std.mem.eql(u8, tag, std.mem.span(scope_ptr))) return true;
        }
        return false;
    }

    /// Scope filter: extrae el tag de `[tag]` al inicio de fmt y verifica
    /// contra el scope_set. Sin scope activo (= count 0) → siempre pasa.
    fn matchesScope(self: Debug, comptime fmt: []const u8) bool {
        const tag = comptime blk: {
            if (fmt.len < 2 or fmt[0] != '[') break :blk "";
            var i: usize = 1;
            while (i < fmt.len and fmt[i] != ']') i += 1;
            if (i >= fmt.len) break :blk "";
            break :blk fmt[1..i];
        };
        if (tag.len == 0) return true; // sin bracket → no filtrar
        return self.scopeMatches(tag);
    }
};

// ─── 4. Breadcrumbs automáticos por etapa ───────────────────────────────────
// Uso:
//   const scope = debugz.BreadcrumbScope.init("pipeline", "prefill");
//   defer scope.exit();
//   // ... código ...
// Emite "[pipeline] > prefill" al entrar y "[pipeline] < prefill (Xms)" al salir.
// Solo emite si DEBUG_LEVEL >= 1 (nivel info).

fn monotonicNs() u64 {
    if (builtin.target.os.tag == .windows) {
        // Windows: use QueryPerformanceCounter for portable monotonic time
        const QPC = struct {
            extern "kernel32" fn QueryPerformanceCounter(*i64) i32;
            extern "kernel32" fn QueryPerformanceFrequency(*i64) i32;
            var freq: ?i64 = null;
            fn getFreq() i64 {
                if (freq) |f| return f;
                var f: i64 = undefined;
                _ = QueryPerformanceFrequency(&f);
                freq = f;
                return f;
            }
        };
        var counter: i64 = undefined;
        _ = QPC.QueryPerformanceCounter(&counter);
        return @intCast(@divTrunc(counter * std.time.ns_per_s, QPC.getFreq()));
    }
    var ts = std.posix.timespec{ .sec = 0, .nsec = 0 };
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub const BreadcrumbScope = struct {
    module: []const u8,
    name: []const u8,
    start_ts: std.Io.Clock.Timestamp,
    active: bool,

    pub fn init(io: std.Io, comptime module: []const u8, comptime name: []const u8) BreadcrumbScope {
        const active = dbg.at(.info);
        if (active) {
            dbg.printLevel(.info, "[{s}] > {s}.{s}\n", .{ module, module, name });
        }
        return .{
            .module = module,
            .name = name,
            .start_ts = if (active) std.Io.Clock.Timestamp.now(io, .awake) else std.Io.Clock.Timestamp{ .raw = std.Io.Timestamp.fromNanoseconds(0), .clock = .awake },
            .active = active,
        };
    }

    pub fn exit(self: *const BreadcrumbScope, io: std.Io) void {
        if (self.active) {
            const elapsed_ns = self.start_ts.untilNow(io).raw.nanoseconds;
            const ms = @as(f64, @floatFromInt(@as(u64, @intCast(elapsed_ns)))) / std.time.ns_per_ms;
            dbg.printLevel(.info, "[{s}] < {s}.{s} ({d:.2}ms)\n", .{ self.module, self.module, self.name, ms });
        }
    }
};

// ─── 6. Memory tracking integrado ──────────────────────────────────────────
// Uso: DEBUG_MEM_SNAPSHOT=/path/to/snapshots
// Cada N calls a snapshot() se emite un breadcrumb con stats de allocación.
// Útil para detectar memory leaks en runs largos sin esperar al exit.

pub const MemorySnapshot = struct {
    path: ?[]const u8,
    interval: u32,
    counter: u32,

    pub fn init() MemorySnapshot {
        const p = blk: {
            const v = std.c.getenv("DEBUG_MEM_SNAPSHOT");
            if (v) |ptr| {
                const s = std.mem.span(ptr);
                if (s.len > 0) break :blk s;
            }
            break :blk null;
        };
        return .{
            .path = p,
            .interval = 100, // cada 100 calls
            .counter = 0,
        };
    }

    pub fn snapshot(self: *MemorySnapshot, label: []const u8) void {
        if (self.path == null) return;
        self.counter += 1;
        if (self.counter % self.interval != 0) return;

        const ts = monotonicNs();
        dbg.printLevel(.info, "[mem_snapshot] {s} at {d}ns (counter={d})\n", .{ label, ts, self.counter });
    }
};

// ─── 7. GPU timing automático ──────────────────────────────────────────────
// Wrapper para cuEvent que facilita el timing automático de kernels CUDA.
// Uso:
//   var timer = try debugz.GpuTimer.start(stream, "my_kernel");
//   // ... kernel launch ...
//   const ms = try timer.stop();
//   dbg.printLevel(.info, "[gpu_kernel] my_kernel: {d:.3}ms\n", .{ms});

pub const GpuTimer = struct {
    stream: ?*anyopaque,
    label: []const u8,
    start_ns: u64,
    is_active: bool,

    pub fn start(stream: ?*anyopaque, comptime label: []const u8) GpuTimer {
        const active = dbg.at(.detail);
        return .{
            .stream = stream,
            .label = label,
            .start_ns = if (active) monotonicNs() else 0,
            .is_active = active,
        };
    }

    pub fn stop(self: *GpuTimer) f64 {
        if (!self.is_active) return 0.0;
        const elapsed_ns = monotonicNs() -| self.start_ns;
        const ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms;
        dbg.printLevel(.detail, "[gpu_timer] {s}: {d:.3}ms\n", .{ self.label, ms });
        return ms;
    }
};

// ─── Helper para auto-timing de secciones ──────────────────────────────────
// Mide el tiempo de ejecución de una función y retorna nanoseconds.
// Uso:
//   const elapsed = debugz.measureNs(struct {
//     fn run() void { /* código a medir */ }
//   }.run);
//   dbg.printLevel(.info, "[perf] took {d:.2}ms\n", .{elapsed / std.time.ns_per_ms});

pub fn measureNs(comptime fn_ptr: anytype) u64 {
    const start = monotonicNs();
    fn_ptr();
    return monotonicNs() -| start;
}

/// Instancia global (se lee de env en `init()`; leerla antes devuelve off).
pub var dbg: Debug = .{};
pub var load_count: usize = 0;

pub fn init() void {
    dbg = Debug.init();
}

fn envOn(name: [:0]const u8) bool {
    return std.c.getenv(name) != null;
}

// ─── Helpers de breadcrumb ──────────────────────────────────────────────────
// Funciones SIEMPRE presentes; el punto de llamada decide con qué flag/nivel
// emitir. Reducen la duplicación de sumas/máximos en los dumps.

pub fn sumAbsF32(slice: []const f32) f64 {
    var s: f64 = 0;
    for (slice) |v| s += @abs(@as(f64, v));
    return s;
}

pub fn maxAbsF32(slice: []const f32) f32 {
    var mx: f32 = 0;
    for (slice) |v| {
        const a = @abs(v);
        if (a > mx) mx = a;
    }
    return mx;
}

/// Suma de valores absolutos de un slice de f16 visto como u16 (KV cache).
pub fn sumAbsF16(u16s: []const u16) f64 {
    var s: f64 = 0;
    for (u16s) |v| s += @abs(@as(f64, @as(f32, @floatFromInt(v))));
    return s;
}

/// 7.2 (lane-f): stats reales sobre f16 (no reinterpretados).
pub fn maxAbsF16Real(slice: []const f16) f32 {
    var mx: f32 = 0;
    for (slice) |v| {
        const a = @abs(@as(f32, @floatCast(v)));
        if (a > mx) mx = a;
    }
    return mx;
}

pub fn sumAbsF16Real(slice: []const f16) f64 {
    var s: f64 = 0;
    for (slice) |v| s += @abs(@as(f64, @as(f32, @floatCast(v))));
    return s;
}

pub fn sumAbsF32Slice(slice: []const f32) f64 {
    return sumAbsF32(slice);
}

pub fn sumAbsF16Slice(u16s: []const u16) f64 {
    return sumAbsF16(u16s);
}
