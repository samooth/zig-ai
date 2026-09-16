//! KVCPT (KV-cache precision tail) — política de cola exacta F16/BF16.
//!
//! Port directo de `beellama.cpp/src/llama-kv-tail-request.{h,cpp}` y la
//! policy function `llama_kvarn_tail_policy_for` en
//! `llama-kv-cache-kvarn.h`. Mantiene la API de surface compatible con
//! `src/main.zig` (--kv-tail-tokens <N|auto|0>, --kv-tail-type <f16|bf16>)
//! y la resolución por grupos (named / positional / numeric / automatic).
//!
//! ## Modelo
//!
//! El cache comprimido (KVarN o Q*) cubre el cuerpo histórico. La cola
//! exacta (últimos N tokens en F16/BF16) se mantiene en un buffer separado
//! (`KvarnExactRing` ya provee esta función) y la atención nativa combina
//! cuerpo + cola sin materializar todo.
//!
//! Para KVarN el grupo es 128 tokens (intrinsic). La policy Bee dice:
//! - `effective_window = 0` → todos exactos (modo compact-native-exact).
//! - Si no, `effective_tokens = max(intrinsic=128, explicit_tokens_redondeado_a_128)`.
//! - `exact_groups = ceil(effective_tokens / 128)`.

const std = @import("std");
const qt = @import("quant_types.zig");

/// Tamaño de grupo KVarN (intrinsic). Duplicado aquí para no importar
/// kvarn.zig y crear dependencias circulares en surface pública.
pub const KVCPT_GROUP: u32 = 128;

/// Default del modo AUTOMATIC upstream (1024 tokens exactos).
pub const KVCPT_AUTOMATIC_DEFAULT: u32 = 1024;

/// Tipo de la cola exacta. Re-export del enum en `quant_types.zig`
/// (duplicado allí para evitar import circular con `tail_request`).
pub const ExactType = qt.ExactType;

/// Modo de la request (transcripción del enum upstream).
pub const TailMode = enum(u8) {
    /// Sin cola exacta (--kv-tail-tokens 0).
    disabled,
    /// Valor numérico único aplicado a todos los grupos (--kv-tail-tokens N).
    numeric,
    /// AUTOMATIC: 1024 tokens exactos por defecto (--kv-tail-tokens auto).
    automatic,
    /// Múltiples valores posicionales, uno por grupo (--kv-tail-tokens 256,512,...).
    positional,
    /// Valores nombrados por group id/role (--kv-tail-tokens group1=256,...).
    named,
};

/// Entrada individual de la request (un grupo nombrado + tokens).
pub const TailEntry = struct {
    /// Nombre del grupo (vacío si la request no es named).
    group: []const u8,
    /// Tokens solicitados (>= 0).
    tokens: u32,
};

/// Request parseada de `--kv-tail-tokens`.
///
/// Inmutable. Se construye con `parse(spec, exact_type)` y se resuelve
/// contra grupos con `resolve(request, groups, kvarn)`.
pub const TailRequest = struct {
    mode: TailMode = .disabled,
    exact_type: ExactType = .default,
    entries: []const TailEntry = &[_]TailEntry{},
    err_msg: ?[]const u8 = null,

    /// `true` si la request es válida y no tiene error.
    pub fn valid(self: TailRequest) bool {
        return self.err_msg == null;
    }
};

/// Resolución de la cola exacta por grupo. Una entrada por grupo del
/// modelo. Equivalente a `llama_kv_tail_request_group_resolution`.
pub const TailGroupResolution = struct {
    /// ID del grupo en el modelo.
    id: []const u8,
    /// Rol del grupo ("" si no aplica).
    role: []const u8,
    /// Tokens solicitados por el usuario (sin resolver).
    raw_requested_tokens: u32,
    /// Tokens después de aplicar el redondeo a múltiplos de grupo.
    requested_tokens: u32,
    /// Tokens efectivos para la cola exacta (min con la ventana del modelo).
    effective_tokens: u32,
    /// Tipo exacto resuelto.
    exact_type: ExactType,
    /// `true` si KVarN sirve la cola desde ring f16 sin materializar.
    native_exact: bool,
};

/// Resultado de resolver una TailRequest contra un conjunto de grupos.
pub const TailResolution = struct {
    /// `true` si la resolución tuvo éxito.
    valid: bool = false,
    /// Resoluciones por grupo (1 por grupo).
    groups: []const TailGroupResolution = &[_]TailGroupResolution{},
    /// Mensaje de error si `valid == false`.
    err_msg: ?[]const u8 = null,
};

/// Descriptor de un grupo del modelo (input para `resolve`).
/// El caller lo construye desde `hparams` + arquitectura.
pub const TailGroup = struct {
    id: []const u8,
    role: []const u8 = "",
    /// Ventana efectiva del grupo (SWA window o context length).
    effective_window: u32,
    /// `true` si el grupo aplica cache KV estándar (no KVarN-only).
    applicable_standard_kv: bool = true,
};

/// Política efectiva de cola exacta para un grupo KVarN (transcripción
/// de `llama_kvarn_tail_policy_for`).
///
/// - `effective_window = 0` → tokens 0, native_exact true (compact-native-exact).
/// - Si no: `intrinsic = min(KVCPT_GROUP, effective_window)`, luego
///   `rounded = ceil(raw / KVCPT_GROUP) * KVCPT_GROUP` (0 si raw=0),
///   `effective_tokens = max(intrinsic, explicit_rounded_clamped_a_window)`.
pub const TailPolicy = struct {
    raw_requested_tokens: u32,
    requested_tokens: u32,
    effective_tokens: u32,
    exact_groups: u32,
    native_exact: bool,
};

pub fn tailPolicyFor(raw_requested_tokens: u32, effective_window: u32) TailPolicy {
    if (effective_window == 0) {
        return .{
            .raw_requested_tokens = raw_requested_tokens,
            .requested_tokens = 0,
            .effective_tokens = 0,
            .exact_groups = 0,
            .native_exact = true,
        };
    }
    const intrinsic: u32 = @min(KVCPT_GROUP, effective_window);
    const rounded: u64 = if (raw_requested_tokens == 0)
        0
    else
        ((@as(u64, raw_requested_tokens) + KVCPT_GROUP - 1) / KVCPT_GROUP) * KVCPT_GROUP;
    const explicit_tokens: u32 = @intCast(@min(rounded, @as(u64, effective_window)));
    const effective_tokens: u32 = @max(intrinsic, explicit_tokens);
    return .{
        .raw_requested_tokens = raw_requested_tokens,
        .requested_tokens = effective_tokens,
        .effective_tokens = effective_tokens,
        .exact_groups = (effective_tokens + KVCPT_GROUP - 1) / KVCPT_GROUP,
        .native_exact = effective_tokens == effective_window,
    };
}

/// Parsea la especificación `--kv-tail-tokens`.
///
/// Formatos:
///   - "" o no presente → disabled
///   - "auto" → automatic (1024 tokens)
///   - "N" → numeric (N tokens todos los grupos)
///   - "N1,N2,..." → positional (1 token por grupo)
///   - "name1=N1,name2=N2,..." → named
///
/// `entries_storage` se usa como buffer para los `TailEntry` resultantes.
/// `error_storage` se usa para mensajes de error (debe ser slice mutable).
/// `name_storage` es el buffer donde se copian los nombres de grupo (de
/// modo que los `TailEntry.group` permanezcan válidos tras retornar).
pub fn parse(
    spec: []const u8,
    exact_type: ExactType,
    entries_storage: []TailEntry,
    name_storage: []u8,
    error_storage: []u8,
) struct {
    request: TailRequest,
    error_written: []const u8,
} {
    var req: TailRequest = .{
        .mode = .disabled,
        .exact_type = exact_type,
    };

    if (exact_type == .bf16) {
        // bf16 soportado para Q* estándar pero no para KVarN native.
        // Aceptamos y dejamos que `resolve` lo rechace si kvarn=true.
    }

    // Trim whitespace.
    const trimmed = std.mem.trim(u8, spec, &[_]u8{ ' ', '\t', ',' });

    if (trimmed.len == 0) {
        req.mode = .disabled;
        return .{ .request = req, .error_written = "" };
    }

    if (std.mem.eql(u8, trimmed, "auto")) {
        req.mode = .automatic;
        return .{ .request = req, .error_written = "" };
    }

    // Split entries por coma.
    var n_entries: usize = 0;
    var entries_iter = std.mem.splitScalar(u8, trimmed, ',');
    var any_named = false;
    var all_named = true;
    var total_entries: usize = 0;
    while (entries_iter.next()) |raw_entry| {
        const entry = std.mem.trim(u8, raw_entry, &[_]u8{ ' ', '\t' });
        if (entry.len == 0) continue;
        total_entries += 1;
        if (std.mem.indexOfScalar(u8, entry, '=') != null) {
            any_named = true;
        } else {
            all_named = false;
        }
    }

    if (any_named and !all_named) {
        const msg = std.fmt.bufPrint(error_storage, "KV tail request cannot mix named and positional groups", .{}) catch "parse error";
        req.err_msg = msg;
        return .{ .request = req, .error_written = msg };
    }

    req.mode = if (any_named)
        .named
    else if (total_entries > 1)
        .positional
    else
        .numeric;

    entries_iter = std.mem.splitScalar(u8, trimmed, ',');
    n_entries = 0;
    var name_buf_len: usize = 0;
    while (entries_iter.next()) |raw_entry| {
        const entry = std.mem.trim(u8, raw_entry, &[_]u8{ ' ', '\t' });
        if (entry.len == 0) continue;
        if (n_entries >= entries_storage.len) {
            const msg = std.fmt.bufPrint(error_storage, "too many KV tail entries (max {d})", .{entries_storage.len}) catch "parse error";
            req.err_msg = msg;
            return .{ .request = req, .error_written = msg };
        }
        const equal_idx = std.mem.indexOfScalar(u8, entry, '=');
        var name_slice: []const u8 = &[_]u8{};
        var value_slice: []const u8 = entry;
        if (equal_idx) |idx| {
            const name_part = std.mem.trim(u8, entry[0..idx], &[_]u8{ ' ', '\t' });
            if (name_part.len == 0) {
                const msg = std.fmt.bufPrint(error_storage, "empty KV tail group name", .{}) catch "parse error";
                req.err_msg = msg;
                return .{ .request = req, .error_written = msg };
            }
            if (name_part.len > name_storage.len - name_buf_len) {
                const msg: []const u8 = "KV tail group name too long";
                @memcpy(error_storage[0..msg.len], msg);
                req.err_msg = error_storage[0..msg.len];
                return .{ .request = req, .error_written = error_storage[0..msg.len] };
            }
            @memcpy(name_storage[name_buf_len..][0..name_part.len], name_part);
            name_slice = name_storage[name_buf_len..][0..name_part.len];
            name_buf_len += name_part.len;
            value_slice = std.mem.trim(u8, entry[idx + 1 ..], &[_]u8{ ' ', '\t' });
        }
        const tokens = parseCount(value_slice) orelse {
            const msg = std.fmt.bufPrint(error_storage, "invalid KV tail token count: {s}", .{value_slice}) catch "parse error";
            req.err_msg = msg;
            return .{ .request = req, .error_written = msg };
        };
        entries_storage[n_entries] = .{ .group = name_slice, .tokens = tokens };
        n_entries += 1;
    }

    req.entries = entries_storage[0..n_entries];
    return .{ .request = req, .error_written = "" };
}

fn parseCount(value: []const u8) ?u32 {
    if (value.len == 0) return null;
    for (value) |c| if (c < '0' or c > '9') return null;
    var result: u64 = 0;
    for (value) |c| {
        const digit: u64 = @intCast(c - '0');
        const next = result * 10 + digit;
        if (next > std.math.maxInt(u32)) return null;
        result = next;
    }
    return @intCast(result);
}

/// Resuelve una TailRequest parseada contra los grupos del modelo.
///
/// `resolutions_storage` se usa para construir las `TailGroupResolution`
/// resultantes. El caller lo mantiene vivo durante el uso de la
/// resolución.
pub fn resolve(
    request: TailRequest,
    groups: []const TailGroup,
    kvarn: bool,
    resolutions_storage: []TailGroupResolution,
) TailResolution {
    var result: TailResolution = .{};
    if (!request.valid()) {
        result.err_msg = if (request.err_msg) |e| e else "invalid KV tail request";
        return result;
    }
    if (resolutions_storage.len < groups.len) return result;

    var raw: []u32 = undefined;
    // Buffer temporal para `raw` por grupo.
    var raw_buf: [16]u32 = [_]u32{0} ** 16;
    if (groups.len <= raw_buf.len) {
        raw = raw_buf[0..groups.len];
    } else {
        // Grupos > 16: usamos el resolutions_storage reusando el campo
        // `requested_tokens` como scratch (no concurrencia, single-threaded).
        raw = undefined;
        for (groups, 0..) |_, i| raw[i] = 0;
        // No hay buffer suficiente; abortar.
        result.err_msg = "too many KV tail groups (max 16)";
        return result;
    }
    @memset(raw, 0);

    switch (request.mode) {
        .disabled => {},
        .numeric => {
            if (request.entries.len != 1) {
                result.err_msg = "numeric KV tail request has no value";
                return result;
            }
            @memset(raw, request.entries[0].tokens);
        },
        .automatic => {
            @memset(raw, KVCPT_AUTOMATIC_DEFAULT);
        },
        .positional => {
            if (request.entries.len != groups.len) {
                result.err_msg = "KV tail positional group count mismatch";
                return result;
            }
            for (request.entries, 0..) |e, i| raw[i] = e.tokens;
        },
        .named => {
            var assigned = [_]bool{false} ** 16;
            @memset(&assigned, false);
            for (request.entries) |entry| {
                var matches: usize = 0;
                var match_idx: usize = 0;
                for (groups, 0..) |g, i| {
                    if (std.mem.eql(u8, g.id, entry.group) or std.mem.eql(u8, g.role, entry.group)) {
                        matches += 1;
                        match_idx = i;
                    }
                }
                if (matches == 0) {
                    result.err_msg = "unknown KV tail group";
                    return result;
                }
                if (matches > 1) {
                    result.err_msg = "ambiguous KV tail group alias";
                    return result;
                }
                if (assigned[match_idx]) {
                    result.err_msg = "duplicate KV tail group assignment";
                    return result;
                }
                raw[match_idx] = entry.tokens;
                assigned[match_idx] = true;
            }
            for (assigned[0..groups.len]) |a| {
                if (!a) {
                    result.err_msg = "incomplete KV tail group configuration";
                    return result;
                }
            }
        },
    }

    const resolved_type: ExactType = if (request.exact_type != .default)
        request.exact_type
    else if (kvarn)
        .f16
    else
        .bf16;

    for (groups, 0..) |g, i| {
        const requested_raw = raw[i];
        var effective: u32 = if (g.applicable_standard_kv)
            @min(requested_raw, g.effective_window)
        else
            0;
        var requested_final: u32 = requested_raw;
        var native_exact = false;
        if (kvarn) {
            const policy = tailPolicyFor(requested_raw, g.effective_window);
            requested_final = policy.requested_tokens;
            effective = policy.effective_tokens;
            native_exact = policy.native_exact;
        }
        resolutions_storage[i] = .{
            .id = g.id,
            .role = g.role,
            .raw_requested_tokens = requested_raw,
            .requested_tokens = requested_final,
            .effective_tokens = effective,
            .exact_type = resolved_type,
            .native_exact = native_exact,
        };
    }

    result.valid = true;
    result.groups = resolutions_storage[0..groups.len];
    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "tailPolicyFor basic cases" {
    // effective_window = 0 → tokens 0, native_exact true
    const p0 = tailPolicyFor(0, 0);
    try std.testing.expectEqual(@as(u32, 0), p0.effective_tokens);
    try std.testing.expect(p0.native_exact);

    // window pequeña (128) con raw grande: max(intrinsic=128, raw redondeado a 128)
    const p1 = tailPolicyFor(512, 128);
    try std.testing.expectEqual(@as(u32, 128), p1.effective_tokens); // clamp a window
    try std.testing.expect(p1.native_exact);

    // window grande (4096), raw 1024 → 1024
    const p2 = tailPolicyFor(1024, 4096);
    try std.testing.expectEqual(@as(u32, 1024), p2.effective_tokens);
    try std.testing.expect(!p2.native_exact);

    // raw redondeado a múltiplos de 128
    const p3 = tailPolicyFor(300, 4096);
    // ceil(300/128) * 128 = 384
    try std.testing.expectEqual(@as(u32, 384), p3.effective_tokens);
}

test "parse disabled / empty / auto" {
    var entries: [8]TailEntry = undefined;
    var err_buf: [256]u8 = undefined;
    var name_buf: [256]u8 = undefined;

    const r0 = parse("", .default, &entries, &err_buf, &name_buf);
    try std.testing.expectEqual(TailMode.disabled, r0.request.mode);

    const r1 = parse("auto", .default, &entries, &err_buf, &name_buf);
    try std.testing.expectEqual(TailMode.automatic, r1.request.mode);

    const r2 = parse("  ", .default, &entries, &err_buf, &name_buf);
    try std.testing.expectEqual(TailMode.disabled, r2.request.mode);
}

test "parse numeric single value" {
    var entries: [8]TailEntry = undefined;
    var err_buf: [256]u8 = undefined;
    var name_buf: [256]u8 = undefined;
    const r = parse("256", .default, &entries, &err_buf, &name_buf);
    try std.testing.expect(r.request.valid());
    try std.testing.expectEqual(TailMode.numeric, r.request.mode);
    try std.testing.expectEqual(@as(usize, 1), r.request.entries.len);
    try std.testing.expectEqual(@as(u32, 256), r.request.entries[0].tokens);
}

test "parse positional multi-value" {
    var entries: [8]TailEntry = undefined;
    var err_buf: [256]u8 = undefined;
    var name_buf: [256]u8 = undefined;
    const r = parse("256,512,1024", .default, &entries, &err_buf, &name_buf);
    try std.testing.expect(r.request.valid());
    try std.testing.expectEqual(TailMode.positional, r.request.mode);
    try std.testing.expectEqual(@as(usize, 3), r.request.entries.len);
    try std.testing.expectEqual(@as(u32, 1024), r.request.entries[2].tokens);
}

test "parse named groups" {
    var entries: [8]TailEntry = undefined;
    var err_buf: [256]u8 = undefined;
    var name_buf: [256]u8 = undefined;
    const r = parse("attn=512,ssm=256", .default, &entries, &err_buf, &name_buf);
    try std.testing.expect(r.request.valid());
    try std.testing.expectEqual(TailMode.named, r.request.mode);
    try std.testing.expectEqualStrings("attn", r.request.entries[0].group);
    try std.testing.expectEqual(@as(u32, 512), r.request.entries[0].tokens);
}

test "parse rejects mixed named/positional" {
    var entries: [8]TailEntry = undefined;
    var err_buf: [256]u8 = undefined;
    var name_buf: [256]u8 = undefined;
    const r = parse("256,attn=512", .default, &entries, &err_buf, &name_buf);
    try std.testing.expect(!r.request.valid());
}

test "resolve numeric on multiple groups" {
    var entries: [4]TailEntry = undefined;
    var err_buf: [256]u8 = undefined;
    var name_buf: [256]u8 = undefined;
    const r = parse("1024", .f16, &entries, &err_buf, &name_buf);
    try std.testing.expect(r.request.valid());

    const groups = [_]TailGroup{
        .{ .id = "layer0", .effective_window = 4096 },
        .{ .id = "layer1", .effective_window = 4096 },
    };
    var resolutions: [2]TailGroupResolution = undefined;
    const res = resolve(r.request, &groups, false, &resolutions);
    try std.testing.expect(res.valid);
    try std.testing.expectEqual(@as(u32, 1024), res.groups[0].effective_tokens);
    try std.testing.expectEqual(ExactType.f16, res.groups[0].exact_type);
}

test "resolve auto gives 1024 tokens for non-kvarn" {
    var entries: [4]TailEntry = undefined;
    var err_buf: [256]u8 = undefined;
    var name_buf: [256]u8 = undefined;
    const r = parse("auto", .default, &entries, &err_buf, &name_buf);
    try std.testing.expect(r.request.valid());

    const groups = [_]TailGroup{
        .{ .id = "g0", .effective_window = 8192 },
    };
    var resolutions: [1]TailGroupResolution = undefined;
    const res = resolve(r.request, &groups, false, &resolutions);
    try std.testing.expect(res.valid);
    // No-kvarn: effective = min(1024, window) = 1024
    try std.testing.expectEqual(@as(u32, 1024), res.groups[0].effective_tokens);
    // bf16 default para non-kvarn.
    try std.testing.expectEqual(ExactType.bf16, res.groups[0].exact_type);
}

test "resolve kvarn applies tailPolicyFor" {
    var entries: [4]TailEntry = undefined;
    var err_buf: [256]u8 = undefined;
    var name_buf: [256]u8 = undefined;
    const r = parse("300", .default, &entries, &err_buf, &name_buf);
    try std.testing.expect(r.request.valid());

    const groups = [_]TailGroup{
        .{ .id = "g0", .effective_window = 4096 },
    };
    var resolutions: [1]TailGroupResolution = undefined;
    const res = resolve(r.request, &groups, true, &resolutions);
    try std.testing.expect(res.valid);
    // KVarN policy: ceil(300/128)*128 = 384, min(384, 4096) = 384.
    try std.testing.expectEqual(@as(u32, 384), res.groups[0].effective_tokens);
    // f16 default para kvarn.
    try std.testing.expectEqual(ExactType.f16, res.groups[0].exact_type);
    try std.testing.expect(!res.groups[0].native_exact); // 384 != 4096
}