//! Tests MoE lane-e:
//!   - E1: parser/stacking GGUF — casos sintéticos contra src/loader/gguf_moe.zig
//!     (builder GGUF en memoria, patrón writeFake* de model_config.zig) + smoke
//!     contra GGUF real vía GGUF_MODEL_PATH.
//!   - E2: offload cache LRU + espejo CPU bit-exacto (próxima tarea).
const std = @import("std");

fn stdoutPrint(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    var stdout_buf: [256]u8 = undefined;
    const stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    try stdout_writer.interface.print(fmt, args);
}
const gguf = @import("gguf");
const moe = @import("gguf_moe");
const budget = @import("budget");

const MoeError = moe.MoeError;
const MoeFamily = moe.MoeFamily;
const testing = std.testing;

const TestWriter = struct {
    buf: std.ArrayList(u8) = .empty,
    n_tensors: u64 = 0,
    n_kvs: u64 = 0,
    data_off: usize = 0,

    fn deinit(self: *TestWriter, alloc: std.mem.Allocator) void {
        self.buf.deinit(alloc);
    }

    fn put(self: *TestWriter, alloc: std.mem.Allocator, bytes: []const u8) !void {
        try self.buf.appendSlice(alloc, bytes);
    }

    fn putStr(self: *TestWriter, alloc: std.mem.Allocator, s: []const u8) !void {
        var hdr: [8]u8 = undefined;
        std.mem.writeInt(u64, &hdr, s.len, .little);
        try self.put(alloc, &hdr);
        try self.put(alloc, s);
    }

    fn kvBegin(self: *TestWriter, alloc: std.mem.Allocator, name: []const u8, vtype: u32) !void {
        self.n_kvs += 1;
        try self.putStr(alloc, name);
        var t: [4]u8 = undefined;
        std.mem.writeInt(u32, &t, vtype, .little);
        try self.put(alloc, &t);
    }

    fn kvStr(self: *TestWriter, alloc: std.mem.Allocator, name: []const u8, val: []const u8) !void {
        try self.kvBegin(alloc, name, 8);
        try self.putStr(alloc, val);
    }

    fn kvU64(self: *TestWriter, alloc: std.mem.Allocator, name: []const u8, val: u64) !void {
        try self.kvBegin(alloc, name, 10);
        var b: [8]u8 = undefined;
        std.mem.writeInt(u64, &b, val, .little);
        try self.put(alloc, &b);
    }

    fn tensorInfo(self: *TestWriter, alloc: std.mem.Allocator, name: []const u8, dims: []const u64, dtype: u32, offset: u64) !void {
        self.n_tensors += 1;
        try self.putStr(alloc, name);
        var b4: [4]u8 = undefined;
        std.mem.writeInt(u32, &b4, @intCast(dims.len), .little);
        try self.put(alloc, &b4);
        for (dims) |d| {
            var b8: [8]u8 = undefined;
            std.mem.writeInt(u64, &b8, d, .little);
            try self.put(alloc, &b8);
        }
        std.mem.writeInt(u32, &b4, dtype, .little);
        try self.put(alloc, &b4);
        var b8: [8]u8 = undefined;
        std.mem.writeInt(u64, &b8, offset, .little);
        try self.put(alloc, &b8);
    }

    /// Cierra cabecera: parchea counts, alinea a 32 y devuelve offset donde
    /// empieza tensor_data (los blobs deben escribirse después con putData).
    fn finishHeader(self: *TestWriter, alloc: std.mem.Allocator) !usize {
        const counts_pos = 8; // magic+version
        std.mem.writeInt(u64, self.buf.items[counts_pos..][0..8], self.n_tensors, .little);
        std.mem.writeInt(u64, self.buf.items[counts_pos + 8 ..][0..8], self.n_kvs, .little);
        while (self.buf.items.len % 32 != 0) try self.buf.append(alloc, 0);
        self.data_off = self.buf.items.len;
        return self.data_off;
    }
};

const Q8_0_BS: u64 = 32;
const Q8_0_BB: u64 = 34;
const F32_T: u32 = 0;
const Q8_0_T: u32 = 8;

fn q80Bytes(nelem: u64) u64 {
    return (nelem / Q8_0_BS) * Q8_0_BB;
}

const BuilderOpts = struct {
    arch: []const u8 = "qwen2moe",
    /// null = modelo denso (sin router)
    layers: []const LayerKind = &.{},
    expert_count: u64 = 4,
    expert_used_count: u64 = 2,
    split: bool = false,
    /// Hueco artificial antes del experto 2 del banco gate (test no-contiguo)
    gap_before_e2: bool = false,
    /// Dimensión errónea en down (test mismatch)
    corrupt_down_dims: bool = false,
    /// Ticket 4.4: escribe `blk.{il}.ffn_down_exps.scale` f32 [E] con
    /// escalas conocidas (0.5 para e=0, 2.0 resto) ⇒ down.external_scale.
    down_external_scale: bool = false,
};

const LayerKind = enum { moe };

const IN_DIM: u64 = 64; // n_embd
const FF_DIM: u64 = 32; // intermediate por experto
const ROUTER_BYTES: u64 = IN_DIM * 4 * 4; // f32 [n_embd, E] con E=4

/// Construye un GGUF sintético. Cada experto del banco `kind` se rellena con
/// el tag `(capa*100 + kind_idx*10 + experto)` repetido; el router con 0xAA.
fn buildGguf(alloc: std.mem.Allocator, opts: BuilderOpts) ![]u8 {
    var w: TestWriter = .{};
    errdefer w.deinit(alloc);

    // Header (counts parcheados en finishHeader)
    try w.put(alloc, "GGUF");
    var b4: [4]u8 = undefined;
    std.mem.writeInt(u32, &b4, 3, .little);
    try w.put(alloc, &b4);
    var b8: [8]u8 = undefined;
    std.mem.writeInt(u64, &b8, 0, .little); // tensor_count (parchea)
    try w.put(alloc, &b8);
    std.mem.writeInt(u64, &b8, 0, .little); // kv_count (parchea)
    try w.put(alloc, &b8);

    try w.kvStr(alloc, "general.architecture", opts.arch);
    var kb: [160]u8 = undefined;
    const k_block_count = std.fmt.bufPrint(&kb, "{s}.block_count", .{opts.arch}) catch unreachable;
    try w.kvU64(alloc, k_block_count, opts.layers.len);
    const k_emb = std.fmt.bufPrint(&kb, "{s}.embedding_length", .{opts.arch}) catch unreachable;
    try w.kvU64(alloc, k_emb, IN_DIM);
    const k_ec = std.fmt.bufPrint(&kb, "{s}.expert_count", .{opts.arch}) catch unreachable;
    try w.kvU64(alloc, k_ec, opts.expert_count);
    const k_uk = std.fmt.bufPrint(&kb, "{s}.expert_used_count", .{opts.arch}) catch unreachable;
    try w.kvU64(alloc, k_uk, opts.expert_used_count);

    // Planificación de offsets (relativos a tensor_data, alineados a 32)
    const kinds = [_][]const u8{ "gate", "up", "down" };
    var offs: std.ArrayList(u64) = .empty;
    defer offs.deinit(alloc);
    var cur: u64 = 0;

    const E = opts.expert_count;
    const per_expert_gate = q80Bytes(IN_DIM * FF_DIM); // ne=[64,32] por experto
    var down_in = FF_DIM;
    const down_out = IN_DIM;
    if (opts.corrupt_down_dims) {
        down_in = FF_DIM + 16;
    }
    const per_expert_down = q80Bytes(down_in * down_out);

    for (opts.layers, 0..) |_, il| {
        var nb: [96]u8 = undefined;
        // router f32 [IN_DIM, E]
        const rnm = std.fmt.bufPrint(&nb, "blk.{d}.ffn_gate_inp.weight", .{il}) catch unreachable;
        try w.tensorInfo(alloc, rnm, &[_]u64{ IN_DIM, E }, F32_T, cur);
        try offs.append(alloc, cur);
        cur += ROUTER_BYTES;
        if (opts.split) {
            for (kinds, 0..) |kd, ki| {
                const eb: u64 = if (ki == 2) per_expert_down else per_expert_gate;
                for (0..@intCast(E)) |e| {
                    const nm = std.fmt.bufPrint(&nb, "blk.{d}.ffn_{s}.experts.{d}.weight", .{ il, kd, e }) catch unreachable;
                    try w.tensorInfo(alloc, nm, &[_]u64{ if (ki == 2) down_in else IN_DIM, if (ki == 2) down_out else FF_DIM }, Q8_0_T, cur);
                    try offs.append(alloc, cur);
                    cur += eb;
                    if (opts.gap_before_e2 and ki == 0 and e == 1) cur += 64;
                }
            }
        } else {
            const dims_gate = [_]u64{ IN_DIM, FF_DIM, E };
            const dims_down = [_]u64{ down_in, down_out, E };
            for (kinds, 0..) |kd, ki| {
                const nm = std.fmt.bufPrint(&nb, "blk.{d}.ffn_{s}_exps.weight", .{ il, kd }) catch unreachable;
                const eb_total: u64 = if (ki == 2) per_expert_down * E else per_expert_gate * E;
                try w.tensorInfo(alloc, nm, if (ki == 2) &dims_down else &dims_gate, Q8_0_T, cur);
                try offs.append(alloc, cur);
                cur += eb_total;
            }
            if (opts.down_external_scale) {
                const snm = std.fmt.bufPrint(&nb, "blk.{d}.ffn_down_exps.scale", .{il}) catch unreachable;
                try w.tensorInfo(alloc, snm, &[_]u64{E}, F32_T, cur);
                try offs.append(alloc, cur);
                cur += E * 4;
            }
        }
    }

    _ = try w.finishHeader(alloc);
    const td = w.data_off;
    try w.buf.resize(alloc, td + @as(usize, @intCast(cur)));

    // Rellenar datos con tags reconocibles
    const data = w.buf.items[td..];
    @memset(data, 0);
    var blob_i: usize = 0;
    for (opts.layers, 0..) |_, il| {
        // router 0xAA
        @memset(data[@intCast(offs.items[blob_i])..][0..@intCast(ROUTER_BYTES)], 0xAA);
        blob_i += 1;
        if (opts.split) {
            for (0..3) |ki| {
                const eb: u64 = if (ki == 2) per_expert_down else per_expert_gate;
                for (0..@intCast(E)) |e| {
                    const tag: u8 = @truncate(il * 100 + ki * 10 + e + 1);
                    const base = offs.items[blob_i];
                    @memset(data[@intCast(base)..][0..@intCast(eb)], tag);
                    blob_i += 1;
                }
            }
        } else {
            for (0..3) |ki| {
                const eb: u64 = if (ki == 2) per_expert_down else per_expert_gate;
                for (0..@intCast(E)) |e| {
                    const tag: u8 = @truncate(il * 100 + ki * 10 + e + 1);
                    const base = offs.items[blob_i] + e * eb;
                    @memset(data[@intCast(base)..][0..@intCast(eb)], tag);
                }
                blob_i += 1;
            }
            if (opts.down_external_scale) {
                // f32 [E]: e=0 → 0.5, resto → 2.0 (verificables en el test 4.4)
                const sbase = offs.items[blob_i];
                for (0..@intCast(E)) |e| {
                    const v: f32 = if (e == 0) 0.5 else 2.0;
                    std.mem.writeInt(u32, data[@intCast(sbase + e * 4)..][0..4], @bitCast(v), .little);
                }
                blob_i += 1;
            }
        }
    }

    return w.buf.toOwnedSlice(alloc);
}

test "denso sin router: no es MoE" {
    const alloc = testing.allocator;
    const raw = try buildGguf(alloc, .{ .arch = "qwen35", .layers = &.{} });
    defer alloc.free(raw);
    var g = try gguf.GgufFile.fromBytes(alloc, raw);
    defer g.deinit();
    try testing.expect(!moe.isMoeModel(&g));
    try testing.expectError(moe.MoeError.RouterNotFound, moe.moeInfo(&g));
}

test "4.4: down con escalas EXTERNAS — parse + composeCanonical canónico" {
    const alloc = testing.allocator;
    const raw = try buildGguf(alloc, .{ .layers = &.{.moe}, .down_external_scale = true });
    defer alloc.free(raw);
    var g = try gguf.GgufFile.fromBytes(alloc, raw);
    defer g.deinit();

    const spec = try moe.layerSpec(&g, 0);
    // Parse: down.external_scale presente y bien formado
    const scales = spec.down.external_scale orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 4), scales.len);
    try testing.expectEqual(@as(f32, 0.5), scales[0]);
    try testing.expectEqual(@as(f32, 2.0), scales[1]);
    // gate/up SIN tensor .scale ⇒ null (canónicos inline)
    try testing.expect(spec.gate.external_scale == null);
    try testing.expect(spec.up.external_scale == null);

    // Compose: banco canónico con d' = d · scale[e]. Bloque q8_0 = [d f16][32 i8].
    // El builder rellena pesos con tag (d f16 @0 es bits del tag). Verificamos
    // que el d de cada bloque del experto e quedó multiplicado (y el resto intacto).
    const composed = try spec.down.composeCanonical(alloc);
    defer alloc.free(composed);
    const bs = spec.down.blockBytes();
    const rb = spec.down.rowBytes();
    for (0..4) |e| {
        const s: f32 = if (e == 0) 0.5 else 2.0;
        const orig_blk = spec.down.expertSlice(e)[0..@intCast(rb)];
        const comp_blk = composed[e * bs ..][0..@intCast(rb)];
        // primer bloque: d escalado
        const d0: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, orig_blk[0..2], .little))));
        const d0c: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, comp_blk[0..2], .little))));
        try testing.expectApproxEqAbs(d0 * s, d0c, 0.01);
        // quanta intactos (solo la escala cambia, NO los i8)
        try testing.expectEqualSlices(u8, orig_blk[2..], comp_blk[2..]);
        // segundo bloque del experto (verificación extra del stride)
        if (rb > 34) {
            const d1: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, orig_blk[34..36], .little))));
            const d1c: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, comp_blk[34..36], .little))));
            try testing.expectApproxEqAbs(d1 * s, d1c, 0.01);
        }
    }
    // Sin .scale ⇒ error claro (gate es canónico)
    if (spec.gate.composeCanonical(alloc)) |_| return error.TestUnexpectedResult else |err| try testing.expectEqual(@as(anyerror, error.NoExternalScales), err);
}

test "qwen2moe fusionado: detecta, apila y lee cero-copia" {
    const alloc = testing.allocator;
    const raw = try buildGguf(alloc, .{ .layers = &.{ .moe, .moe } });
    defer alloc.free(raw);
    var g = try gguf.GgufFile.fromBytes(alloc, raw);
    defer g.deinit();

    try testing.expect(moe.isMoeModel(&g));
    const info = try moe.moeInfo(&g);
    try testing.expectEqual(moe.MoeFamily.qwen2_moe, info.family);
    try testing.expectEqual(@as(u32, 4), info.n_expert);
    try testing.expectEqual(@as(u32, 2), info.top_k);

    const spec = try moe.layerSpec(&g, 0);
    try testing.expectEqual(@as(u32, 4), spec.n_expert);
    try testing.expectEqual(@as(u32, 2), spec.top_k);
    // router f32 [64,4] lleno de 0xAA
    try testing.expectEqual(gguf.GgmlType.f32, spec.router.dtype);
    try testing.expectEqual(@as(usize, 64), spec.router.n_embd);
    try testing.expectEqual(@as(u8, 0xAA), spec.router.bytes[0]);

    // Banco gate fusionado: 4 expertos × (2048 elems → 64 bloques × 34B)
    const eb: usize = @intCast(q80Bytes(IN_DIM * FF_DIM));
    try testing.expectEqual(@as(usize, eb * 4), spec.gate.bytes.len);
    try testing.expectEqual(gguf.GgmlType.q8_0, spec.gate.dtype);
    try testing.expectEqual(@as(usize, 64), spec.gate.in_dim);
    try testing.expectEqual(@as(usize, 32), spec.gate.out_dim);
    try testing.expectEqual(eb / 32, spec.gate.rowBytes());

    // Zero-copy: el slice vive dentro de la copia propia del parser y cada
    // experto lleva su tag.
    for (0..4) |e| {
        const sl = spec.gate.expertSlice(e);
        try testing.expectEqual(@as(u8, @truncate(0 * 100 + 0 * 10 + e + 1)), sl[0]);
        try testing.expectEqual(sl[0], sl[sl.len - 1]);
    }
    // down: ne=[32,64] → out_dim 64, row = 2176/64 = 34B
    try testing.expectEqual(@as(usize, 64), spec.down.out_dim);
    try testing.expectEqual(@as(usize, 34), spec.down.rowBytes());
    // Capa 1 independiente (tags 1xx)
    const spec1 = try moe.layerSpec(&g, 1);
    try testing.expectEqual(@as(u8, @truncate(1 * 100 + 2 * 10 + 3 + 1)), spec1.down.expertSlice(3)[0]);
    try testing.expect(!moe.isMoeLayer(&g, 2));
}

test "gpt-oss split: stacking contiguo de 4 expertos" {
    const alloc = testing.allocator;
    const raw = try buildGguf(alloc, .{
        .arch = "gpt-oss",
        .layers = &.{.moe},
        .split = true,
    });
    defer alloc.free(raw);
    var g = try gguf.GgufFile.fromBytes(alloc, raw);
    defer g.deinit();

    const info = try moe.moeInfo(&g);
    try testing.expectEqual(moe.MoeFamily.gpt_oss, info.family);

    const spec = try moe.layerSpec(&g, 0);
    const eb: usize = @intCast(q80Bytes(IN_DIM * FF_DIM));
    try testing.expectEqual(@as(usize, eb * 4), spec.up.bytes.len);
    for (0..4) |e| {
        const sl = spec.gate.expertSlice(e);
        try testing.expectEqual(@as(u8, @truncate(0 * 100 + 0 * 10 + e + 1)), sl[0]);
        try testing.expectEqual(sl[0], sl[sl.len - 1]);
    }
}

test "split con hueco: error ExpertsNotContiguous" {
    const alloc = testing.allocator;
    const raw = try buildGguf(alloc, .{
        .arch = "gpt-oss",
        .layers = &.{.moe},
        .split = true,
        .gap_before_e2 = true,
    });
    defer alloc.free(raw);
    var g = try gguf.GgufFile.fromBytes(alloc, raw);
    defer g.deinit();

    // El router y la metadata están bien; el banco gate tiene un hueco.
    try testing.expect(moe.isMoeModel(&g));
    try testing.expectError(moe.MoeError.ExpertsNotContiguous, moe.layerSpec(&g, 0));
}

test "dims corruptas en down: error ExpertDimMismatch" {
    const alloc = testing.allocator;
    const raw = try buildGguf(alloc, .{ .layers = &.{.moe}, .corrupt_down_dims = true });
    defer alloc.free(raw);
    var g = try gguf.GgufFile.fromBytes(alloc, raw);
    defer g.deinit();

    try testing.expectError(moe.MoeError.ExpertDimMismatch, moe.layerSpec(&g, 0));
}

test "smoke contra GGUF real (GGUF_MODEL_PATH)" {
    const env_path = std.c.getenv("GGUF_MODEL_PATH") orelse return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    var g = try gguf.GgufFile.fromFileMmap(io, testing.allocator, std.mem.span(env_path));
    defer g.deinit();

    if (!moe.isMoeModel(&g)) {
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[gguf_moe] {s}: denso (sin router)\n", .{"modelo"});
        return;
    }
    const info = try moe.moeInfo(&g);
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[gguf_moe] real: family={s} E={d} top_k={d}\n", .{ @tagName(info.family), info.n_expert, info.top_k });
    // Recorre todas las capas buscando specs válidas (no asserts de valores:
    // el objetivo es ejercitar el parser sobre bytes reales).
    const bc = moe.blockCountMeta(&g, g.arch().?) orelse 0;
    var n_moe: usize = 0;
    for (0..@intCast(bc)) |il| {
        if (!moe.isMoeLayer(&g, il)) continue;
        n_moe += 1;
        const spec = moe.layerSpec(&g, il) catch |err| {
            try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[gguf_moe] capa {d}: error {s}\n", .{ il, @errorName(err) });
            continue;
        };
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[gguf_moe] capa {d} ok: E={d} row(gate)={d}B row(down)={d}B\n", .{ il, spec.n_expert, spec.gate.rowBytes(), spec.down.rowBytes() });
    }
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[gguf_moe] capas MoE encontradas: {d}/{d}\n", .{ n_moe, bc });
}

// ═══════════════════════════════════════════════════════════════════════════
// E2 — OffloadCache: espejo CPU bit-exacto (oráculo del kernel E3)

const cache = @import("offload_cache");

fn mkCache(alloc: std.mem.Allocator, L: u32, E: u32, C: u32, max_fetch: u32) !cache.OffloadCache {
    return cache.OffloadCache.init(alloc, .{
        .num_layers = L,
        .num_experts = E,
        .cache_size = C,
        .max_fetch = max_fetch,
    });
}

test "E2: cold all-miss con cupo fetchea todo y reescribe a slots" {
    const alloc = testing.allocator;
    var c = try mkCache(alloc, 1, 4, 4, 8);
    defer c.deinit(alloc);

    var ids = [_]i32{ 3, 0, 2 };
    c.ensureExpertsMirror(0, &ids, 0, null);

    // Scores con recencia −1: −E+(E−1−e) ⇒ orden de fetch e0 < e2 < e3.
    try testing.expectEqual(@as(i64, 3), c.num_missing_full);
    try testing.expectEqual(@as(i64, 3), c.num_indices);
    // Víctimas: usage todo 0 ⇒ slots 0,1,2 en orden; ganadores e0,e2,e3.
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 1, 2 }, c.evict_slots[0..3]);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 3 }, c.src_indices[0..3]);
    // Rewrite: id3→slot2, id0→slot0, id2→slot1.
    try testing.expectEqualSlices(i32, &[_]i32{ 2, 0, 1 }, &ids);
    try testing.expectEqual(@as(i32, 0), c.id_of_slot[0]);
    try testing.expectEqual(@as(i32, 3), c.id_of_slot[2]);
    // id plano 1 (layer0,e1) nunca activo ⇒ no residente; id plano 3 → slot2.
    try testing.expectEqual(@as(i32, -1), c.slot_for_id[1]);
    try testing.expectEqual(@as(i32, 2), c.slot_for_id[3]);
    // Todos los slots usados en este paso tienen last_access actualizado.
    try testing.expectEqual(@as(u64, 0), c.last_access[0]);
    try testing.expectEqual(@as(u64, 1), c.last_access[1]);
    try testing.expectEqual(@as(u64, 2), c.last_access[2]);
}

test "E2: hits puros no alteran mapa y bump uso" {
    const alloc = testing.allocator;
    var c = try mkCache(alloc, 1, 4, 4, 8);
    defer c.deinit(alloc);

    var ids = [_]i32{ 1, 3 };
    c.ensureExpertsMirror(0, &ids, 0, null); // step1: fetch 1,3 → slots 0,1
    var ids2 = [_]i32{ 3, 1 };
    const before = c.last_access;
    _ = before;
    c.ensureExpertsMirror(0, &ids2, 0, null); // step2: puro hit
    try testing.expectEqual(@as(i64, 0), c.num_missing_full);
    try testing.expectEqual(@as(i64, 0), c.num_indices);
    // ids reescritos a los mismos slots que el paso anterior (id3→slot1? no:
    // paso1 ganadores por score: e1 primero (score −4+2=−2) slot0, luego e3
    // (−4) slot1 ⇒ id1→slot0, id3→slot1).
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 0 }, &ids2);
    try testing.expectEqual(@as(u64, 2), c.last_access[0]);
    try testing.expectEqual(@as(u64, 3), c.last_access[1]);
}

test "E2: overflow respeta cap fijo y marca −1" {
    const alloc = testing.allocator;
    var c = try mkCache(alloc, 1, 8, 8, 2); // cap duro 2
    defer c.deinit(alloc);

    var ids = [_]i32{ 5, 1, 6, 0, 7 };
    c.ensureExpertsMirror(0, &ids, 0, null);
    try testing.expectEqual(@as(i64, 5), c.num_missing_full);
    try testing.expectEqual(@as(i64, 2), c.num_indices);
    var n_fetched: usize = 0;
    var n_cpu: usize = 0;
    for (ids) |v| {
        if (v >= 0) n_fetched += 1 else n_cpu += 1;
    }
    try testing.expectEqual(@as(usize, 2), n_fetched);
    try testing.expectEqual(@as(usize, 3), n_cpu);
    // Los dos primeros ganadores por recencia/id: e0 y e1 (scores más altos).
    try testing.expectEqual(@as(i32, 0), c.src_indices[0]);
    try testing.expectEqual(@as(i32, 1), c.src_indices[1]);
}

test "E2: Q16 frac=65536 fetchea todo" {
    const alloc = testing.allocator;
    var c = try mkCache(alloc, 1, 4, 4, 1); // cap fijo 1, pero frac manda
    defer c.deinit(alloc);

    var ids = [_]i32{ 0, 2, 3 };
    c.ensureExpertsMirror(0, &ids, 65536, null);
    try testing.expectEqual(@as(i64, 3), c.num_indices);
    for (ids) |v| try testing.expect(v >= 0);
}

test "E2: Q16 tabla de valores esperados" {
    const alloc = testing.allocator;
    var c = try mkCache(alloc, 1, 16, 16, 99);
    defer c.deinit(alloc);

    const cases = [_][3]i64{
        // {num_activos_distintos, frac_q16, fetch_esperado}
        .{ 7, 32768, 3 }, // lo=3; cost_lo=max(3·32768,4·32768)=131072 == cost_hi ⇒ lo
        .{ 1, 1, 0 }, // lo=0; cost_lo=1 <= cost_hi=1 ⇒ 0
        .{ 10, 60000, 10 }, // lo=9; cost_lo=60000 > cost_hi=55360 ⇒ 10
        .{ 16, 4096, 1 }, // lo=1; cost_lo=max(4096,15·4096)=61440 > cost_hi=max(8192,14·4096)=57344 ⇒ 2? NO: ver abajo
        .{ 3, 21845, 1 }, // lo=1; cost_lo=max(43691,2·21845)=43690 vs cost_hi=max(2·43691,21845)=87382 ⇒ lo=1
    };
    // Caso {16,4096}: lo=(16·4096)>>16=1; cost_lo=max(1·61440, 15·4096)=61440;
    // cost_hi=max(2·61440, 14·4096)=122880 ⇒ cost_lo<=cost_hi ⇒ fetch=lo=1.
    for (cases) |tc| {
        c.reset();
        var ids: [16]i32 = undefined;
        for (0..@intCast(tc[0])) |k| ids[k] = @intCast(k);
        const slice = ids[0..@intCast(tc[0])];
        c.step = 100; // recencia base distinta para forzar all-miss limpio
        c.ensureExpertsMirror(0, slice, @intCast(tc[1]), null);
        try testing.expectEqual(tc[2], c.num_indices);
    }
}

test "E2: pool compartido — evict global LRU protegiendo activos" {
    const alloc = testing.allocator;
    var c = try mkCache(alloc, 2, 4, 4, 8); // 2 capas, 4 slots compartidos
    defer c.deinit(alloc);

    // Capa 0 llena los 4 slots (step1): ids 0..3 → slots 0..3.
    var ids0 = [_]i32{ 0, 1, 2, 3 };
    c.ensureExpertsMirror(0, &ids0, 0, null);
    // Capa 0 usa solo 0 y 1 (step2): sus usages suben a 2.
    var keep = [_]i32{ 0, 1 };
    c.ensureExpertsMirror(0, &keep, 0, null);

    // Capa 1 pide e1 (miss): debe evictar la víctima LRU global entre slots
    // con usage menor — slots 2,3 quedaron con usage=1 (protegidos NO son:
    // pertenecen a capa 0 pero no activos). argmin primera ocurrencia: slot2.
    var ids1 = [_]i32{1};
    c.ensureExpertsMirror(1, &ids1, 0, null);
    try testing.expectEqual(@as(i32, 2), c.evict_slots[0]);
    try testing.expectEqual(@as(i32, 4 + 1), c.id_of_slot[2]); // id plano capa1·4+1=5
    // La capa 0 perdió su experto 2: slot_for_id[2] ahora -1.
    try testing.expectEqual(@as(i32, -1), c.slot_for_id[0 * 4 + 2]);
}

test "E2: recencia prioriza miss recurrente en el siguiente paso" {
    const alloc = testing.allocator;
    var c = try mkCache(alloc, 1, 8, 4, 4); // pool chico para rotación real
    defer c.deinit(alloc);

    // Paso 1: {0,5} → s0←e0, s1←e5 (usage 1). recency[e0]=recency[e5]=1.
    var a = [_]i32{ 0, 5 };
    c.ensureExpertsMirror(0, &a, 0, null);
    try testing.expect(a[0] >= 0 and a[1] >= 0);

    // Paso 2: llenamos TODO el pool con otros expertos: víctimas = slots
    // vacíos (usage 0) primero (s2,s3), luego los usados (s0,s1).
    // Ganadores por id asc: e2,e3,e6,e7 ⇒ s2←e2, s3←e3, s0←e6, s1←e7.
    // e0 y e5 quedan EVICTADOS; su recencia NO se toca (no eran activos).
    var b = [_]i32{ 2, 6, 3, 7 };
    c.ensureExpertsMirror(0, &b, 0, null);
    for (b) |v| try testing.expect(v >= 0);
    try testing.expectEqual(@as(i32, -1), c.slot_for_id[0]); // e0 fuera

    // Paso 3: activos {0,5,4}, cap 4 pero fetch=min(3 misses... e0/e5 miss
    // (recencia 1), e4 nunca visto (recencia −1): 3 misses, fetch 3.
    // Scores: e0: 1·8+7=15; e5: 8+2=10; e4: −8+3=−5 ⇒ ganadores e0,e5,e4.
    var d = [_]i32{ 0, 5, 4 };
    c.ensureExpertsMirror(0, &d, 0, null);
    try testing.expectEqual(@as(i64, 3), c.num_indices);
    try testing.expectEqual(@as(i32, 0), c.src_indices[0]);
    try testing.expectEqual(@as(i32, 5), c.src_indices[1]);
    try testing.expectEqual(@as(i32, 4), c.src_indices[2]);
    for (d) |v| try testing.expect(v >= 0);
}

test "E2: duplicados en ids se manejan como un solo activo" {
    const alloc = testing.allocator;
    var c = try mkCache(alloc, 1, 4, 4, 8);
    defer c.deinit(alloc);

    var ids = [_]i32{ 2, 2, 2 };
    c.ensureExpertsMirror(0, &ids, 0, null);
    try testing.expectEqual(@as(i64, 1), c.num_missing_full);
    // stat_active acumula ENTRADAS (num_active=len), igual que el kernel.
    try testing.expectEqual(@as(i64, 3), c.stat_active);
    for (ids) |v| try testing.expect(v == ids[0]);
    try testing.expect(ids[0] >= 0);
}

test "E2: stats por capa y rates" {
    const alloc = testing.allocator;
    var c = try mkCache(alloc, 2, 4, 4, 1); // cap 1: siempre queda miss
    defer c.deinit(alloc);

    var la = [_]i32{ 0, 1 };
    c.ensureExpertsMirror(0, &la, 0, null);
    var lb = [_]i32{ 2, 3 };
    c.ensureExpertsMirror(1, &lb, 0, null);

    try testing.expectEqual(@as(i64, 2), c.stat_calls);
    try testing.expectEqual(@as(i64, 4), c.stat_active);
    try testing.expectEqual(@as(i64, 4), c.stat_missing);
    try testing.expectEqual(@as(i64, 2), c.stat_fetched);
    try testing.expectEqual(@as(f64, 0.5), c.fetchRate());
    try testing.expectEqual(@as(i64, 2), c.stat_missing_layer[0]);
    try testing.expectEqual(@as(i64, 1), c.stat_fetched_layer[1]);
    try testing.expectEqual(@as(i64, 1), c.stat_steps_layer[1]);
}

test "E2: ids inválidos pasan a −1 sin crash" {
    const alloc = testing.allocator;
    var c = try mkCache(alloc, 1, 4, 4, 8);
    defer c.deinit(alloc);

    var ids = [_]i32{ -1, 9, 2 };
    c.ensureExpertsMirror(0, &ids, 0, null);
    try testing.expectEqual(@as(i32, -1), ids[0]);
    try testing.expectEqual(@as(i32, -1), ids[1]);
    try testing.expect(ids[2] >= 0);
}

// ── Caso combinado gemma-style: ffn_gate_up_exps [in, 2*ff, E] ─────────────

const CU_IN: u64 = 64;
const CU_FF: u64 = 32; // por mitad ⇒ out_total = 64
const CU_E: u64 = 4;

/// GGUF mínimo con UNA capa cuyo gate+up vive en `ffn_gate_up_exps` (Q8_0,
/// bloque por experto = [mitad gate | mitad up]) y down separado.
fn buildCombinedGguf(alloc: std.mem.Allocator) ![]u8 {
    var w: TestWriter = .{};
    errdefer w.deinit(alloc);

    try w.put(alloc, "GGUF");
    var b4: [4]u8 = undefined;
    std.mem.writeInt(u32, &b4, 3, .little);
    try w.put(alloc, &b4);
    var b8: [8]u8 = .{0} ** 8;
    try w.put(alloc, &b8); // tensors
    try w.put(alloc, &b8); // kvs

    const arch = "gemma4moe";
    try w.kvStr(alloc, "general.architecture", arch);
    try w.kvU64(alloc, arch ++ ".block_count", 1);
    try w.kvU64(alloc, arch ++ ".embedding_length", CU_IN);
    try w.kvU64(alloc, arch ++ ".expert_count", CU_E);
    try w.kvU64(alloc, arch ++ ".expert_used_count", 2);

    var offs: std.ArrayList(u64) = .empty;
    defer offs.deinit(alloc);
    var cur: u64 = 0;

    // router
    var nb: [96]u8 = undefined;
    try w.tensorInfo(alloc, std.fmt.bufPrint(&nb, "blk.{d}.ffn_gate_inp.weight", .{0}) catch unreachable, &[_]u64{ CU_IN, CU_E }, F32_T, cur);
    try offs.append(alloc, cur);
    cur += CU_IN * CU_E * 4;

    // gate_up combinado: ne=[CU_IN, CU_FF*2, CU_E]
    const eb_half: u64 = CU_IN * CU_FF / Q8_0_BS * Q8_0_BB;
    const eb_block: u64 = eb_half * 2;
    try w.tensorInfo(alloc, std.fmt.bufPrint(&nb, "blk.{d}.ffn_gate_up_exps.weight", .{0}) catch unreachable, &[_]u64{ CU_IN, CU_FF * 2, CU_E }, Q8_0_T, cur);
    try offs.append(alloc, cur);
    cur += eb_block * CU_E;

    // down separado ne=[CU_FF, CU_IN, CU_E]
    try w.tensorInfo(alloc, std.fmt.bufPrint(&nb, "blk.{d}.ffn_down_exps.weight", .{0}) catch unreachable, &[_]u64{ CU_FF, CU_IN, CU_E }, Q8_0_T, cur);
    try offs.append(alloc, cur);
    cur += eb_half * CU_E;

    _ = try w.finishHeader(alloc);
    const td = w.data_off;
    try w.buf.resize(alloc, td + @as(usize, @intCast(cur)));
    const data = w.buf.items[td..];
    @memset(data, 0);

    // router 0xAA
    const router_len: usize = @intCast(CU_IN * CU_E * 4);
    @memset(data[@intCast(offs.items[0])..][0..router_len], 0xAA);
    // bloques combinados: mitad gate=tag 0x10+e, mitad up=0x40+e
    for (0..@intCast(CU_E)) |e| {
        const base: usize = @intCast(offs.items[1]);
        const hb: usize = @intCast(eb_half);
        const blk: usize = @intCast(eb_block);
        @memset(data[base + e * blk ..][0..hb], @truncate(0x10 + e));
        @memset(data[base + e * blk + hb ..][0..hb], @truncate(0x40 + e));
    }
    // down: tag 0xDD
    @memset(data[@intCast(offs.items[2])..][0..@intCast(eb_half * CU_E)], 0xDD);

    return w.buf.toOwnedSlice(alloc);
}

test "E1-combinado: gate_up_exps produce ventanas gate/up correctas" {
    const alloc = testing.allocator;
    const raw = try buildCombinedGguf(alloc);
    defer alloc.free(raw);
    var g = try gguf.GgufFile.fromBytes(alloc, raw);
    defer g.deinit();

    try testing.expect(moe.isMoeModel(&g));
    const info = try moe.moeInfo(&g);
    try testing.expectEqual(@as(u32, 4), info.n_expert);
    try testing.expectEqual(@as(u32, 2), info.top_k);

    const spec = try moe.layerSpec(&g, 0);
    // mitades correctas
    try testing.expectEqual(@as(usize, @intCast(CU_FF)), spec.gate.out_dim);
    try testing.expectEqual(@as(usize, @intCast(CU_FF)), spec.up.out_dim);
    const half: usize = @intCast(CU_IN * CU_FF / Q8_0_BS * Q8_0_BB);
    try testing.expectEqual(half, spec.gate.expertBytes());
    try testing.expectEqual(half, spec.up.expertBytes());

    // ventanas: up apunta EXACTAMENTE donde termina la mitad gate del mismo e
    const eb_half_eb: usize = @intCast(CU_IN * CU_FF / Q8_0_BS * Q8_0_BB);
    _ = eb_half_eb;
    for (0..@intCast(CU_E)) |e| {
        const gs = spec.gate.expertSlice(e);
        const us = spec.up.expertSlice(e);
        try testing.expectEqual(@as(usize, half), gs.len);
        try testing.expectEqual(@as(usize, half), us.len);
        try testing.expectEqual(@as(u8, @truncate(0x10 + e)), gs[0]);
        try testing.expectEqual(@as(u8, @truncate(0x40 + e)), us[0]);
        // contigüidad: up empieza justo tras gate dentro del bloque
        try testing.expectEqual(@intFromPtr(gs.ptr) + half, @intFromPtr(us.ptr));
    }
    // down intacto (tag 0xDD)
    try testing.expectEqual(@as(u8, 0xDD), spec.down.expertSlice(0)[0]);
}

test "E2: budget pressure caps fetch and evicts LRU" {
    const alloc = testing.allocator;
    var c = try cache.OffloadCache.init(alloc, .{
        .num_layers = 1,
        .num_experts = 4,
        .cache_size = 4,
        .max_fetch = 8,
        .slot_bytes = 1024,
    });
    defer c.deinit(alloc);

    // Fill cache with 4 experts.
    var ids = [_]i32{ 0, 1, 2, 3 };
    c.ensureExpertsMirror(0, &ids, 0, null);
    try testing.expectEqual(@as(usize, 4 * 1024), c.residentBytes());

    // Evict to budget 2*1024.
    c.evictToBudget(2 * 1024);
    try testing.expect(c.residentBytes() <= 2 * 1024);

    // shrinkableFloor is 0.
    try testing.expectEqual(@as(usize, 0), c.shrinkableFloor());
}

test "E2: budget.Rebuilder round-trip with offload_cache" {
    const alloc = testing.allocator;
    var c = try cache.OffloadCache.init(alloc, .{
        .num_layers = 1,
        .num_experts = 4,
        .cache_size = 4,
        .max_fetch = 8,
        .slot_bytes = 1024,
    });
    defer c.deinit(alloc);

    var ids = [_]i32{ 0, 1, 2, 3 };
    c.ensureExpertsMirror(0, &ids, 0, null);
    try testing.expectEqual(@as(usize, 4 * 1024), c.residentBytes());

    const consumer = c.asConsumer("offload");
    var consumers = [_]budget.Consumer{consumer};
    var rb = budget.Rebuilder.init(.{
        .baseline_free = 0,
        .weights_bytes = 0,
        .fixed_cache = 0,
        .memory_ratio = 1.0,
    }, &consumers);

    // Tight snapshot: net = 0 ⇒ floor (4 KiB) > budget ⇒ overcommitted.
    const tight = budget.BudgetSnapshot{ .baseline_free = 0, .weights_bytes = 0, .fixed_cache = 0, .memory_ratio = 1.0 };
    try testing.expect(rb.fitCheck(tight) == null);

    // Loose snapshot: net = 8 KiB > floor (4 KiB) ⇒ fit; apply updates current.
    const loose = budget.BudgetSnapshot{ .baseline_free = 8192, .weights_bytes = 0, .fixed_cache = 0, .memory_ratio = 1.0 };
    const result = rb.fitCheck(loose);
    try testing.expect(result != null);
    try testing.expectEqual(budget.PlanResult.fit, result.?);
    const gen_before = rb.generation;
    rb.apply(loose);
    try testing.expectEqual(gen_before + 1, rb.generation);
    try testing.expectEqual(@as(usize, 4 * 1024), consumer.current);

    // Simulate alloc failure: pre‑validate a bad ratio, then rollback.
    try testing.expect(!budget.Rebuilder.preValidate(.{ .baseline_free = 1000, .weights_bytes = 0, .fixed_cache = 0, .memory_ratio = 0.0 }));
    rb.rollback();
    try testing.expectEqual(@as(usize, 4 * 1024), consumer.current);
}
