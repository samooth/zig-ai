//! Paso 1 KV-Codec §A1 (lane-kvc): trazas K/V del forward CPU.
//! Valida el roundtrip appendChunk→finish→bin: layout [T,n_kv_head,head_dim]
//! f16 LE y manifest. No requiere GPU ni modelo.
const std = @import("std");
const kvc = @import("kv_cache");
const ggufr = @import("gguf");
const model_config = @import("model_config");
const qw_mod = @import("quant_weight");
const KvTrace = kvc.kv_trace.KvTrace;

test "kv_trace append/finish escribe bins con valores correctos" {
    const t = std.testing;
    var arena_state = std.heap.ArenaAllocator.init(t.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.Io.Threaded.global_single_threaded.io();
    var io_mut = io;

    const root = "/tmp/kilo/kv_trace_test";

    var tr = try KvTrace.init(arena, &io_mut, root, "qwen_test", "prose", 2, 2, 4, 4);
    defer tr.deinit();

    // 2 tokens, kv=2 heads × dim 4 ⇒ 8 elems/token
    const n: usize = 2;
    var k_pre: [16]f32 = undefined;
    var k_post: [16]f32 = undefined;
    var v: [16]f32 = undefined;
    var q: [32]f32 = undefined;
    for (0..16) |i| {
        k_pre[i] = @floatFromInt(i);
        k_post[i] = @floatFromInt(i * 2);
        v[i] = @floatFromInt(i * 3);
    }
    for (0..32) |i| q[i] = @as(f32, @floatFromInt(i)) * 0.5;
    try tr.appendChunk(0, n, &k_pre, &k_post, &v, &q);
    try tr.finish();

    // k_pre.bin: 2 tokens × 8 elems × 2 bytes = 32 bytes, valores f16 exactos
    var pbuf: [256]u8 = undefined;
    const kp_path = try std.fmt.bufPrint(&pbuf, "{s}/qwen_test/prose/L0/k_pre.bin", .{root});
    var file = try std.Io.Dir.cwd().openFile(io, kp_path, .{ .mode = .read_only });
    defer file.close(io);
    var rbuf: [32]u8 = undefined;
    const n_read = try file.readPositionalAll(io, &rbuf, 0);
    try t.expectEqual(@as(usize, 32), n_read);
    const halves = std.mem.bytesAsSlice(u16, rbuf[0..]);
    // 3.0 (i=3) debe codificarse como f16 0x4200 (little-endian: 0x00,0x42)
    try t.expectEqual(@as(u16, 0x4200), halves[3]);

    // manifest: T de la primera capa con tokens. El manifest hace append
    // en corridas repetidas: el test busca la línea del corpus "prose".
    const mp_path = try std.fmt.bufPrint(&pbuf, "{s}/qwen_test/prose/manifest.json", .{root});
    var mfile = try std.Io.Dir.cwd().openFile(io, mp_path, .{ .mode = .read_only });
    defer mfile.close(io);
    var mbuf: [512]u8 = undefined;
    const m_read = try mfile.readPositionalAll(io, &mbuf, 0);
    const manifest = mbuf[0..m_read];
    try t.expect(std.mem.indexOf(u8, manifest, "\"T\":2") != null);
    try t.expect(std.mem.indexOf(u8, manifest, "\"corpus\":\"prose\"") != null);
}

test "kv_trace hook: appendChunk vía hook respeta disabled" {
    const t = std.testing;
    var arena_state = std.heap.ArenaAllocator.init(t.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // hook sin tracer: stage/append son no-op silenciosos (coste cero off)
    kvc.kv_trace.hook.setTracer(null, arena);
    try kvc.kv_trace.hook.stageKPre(&.{ 1.0, 2.0 });
    try kvc.kv_trace.hook.stageV(&.{3.0});
    try kvc.kv_trace.hook.appendChunk(0, 1, 2, &.{ 4.0, 4.0 }, &.{5.0});
}

// ═══════════════════════════════════════════════════════════════════════════
// Lane-KVC P1: roundtrip dequant i2_s
// ═══════════════════════════════════════════════════════════════════════════

test "dequant i2_s: roundtrip con fixture sintético" {
    // 4 grupos de 128 valores: todos cero
    const bytes = [_]u8{0} ** (4 * 32 + 4);
    // scale = 0 → todo dequant = 0
    const scale_zero: [4]u8 = .{ 0, 0, 0, 0 };
    const full = bytes ++ scale_zero;
    var out: [4 * 128]f32 = undefined;
    ggufr.dequantI2_S(&full, &out);
    for (out) |v| try std.testing.expectEqual(@as(f32, 0.0), v);

    // 4 grupos todos +1, scale = 1.0
    var ones_buf: [132]u8 = undefined;
    for (0..128) |i| ones_buf[i] = 0xAA;
    @memcpy(ones_buf[128..][0..4], &[_]u8{ 0, 0, 0x80, 0x3F }); // scale f32 1.0 LE
    var out2: [4 * 128]f32 = undefined;
    ggufr.dequantI2_S(&ones_buf, &out2);
    // q=2 → +1.0 en la mitad de los valores (bits 1 de cada par: posición 1,3,5,7 en byte)
    var count_pos: usize = 0;
    for (out2) |v| {
        if (v == 1.0) count_pos += 1;
    }
    try std.testing.expect(count_pos > 0); // sanity
}

test "dequant i2_s: layout del GGUF real (primer attn_q de bitnet-b1.58-2B)" {
    // Carga el primer peso del modelo real y verifica:
    // (1) scale plausible [0.5, 3.0]
    // (2) pesos dequant en rango ±scale
    // (3) fracciones de {0, ±1} ~50% cada uno (modelo ternario entrenado)
    const path = "/ai/models/bitnet/ggml-model-i2_s.gguf";
    var file = std.Io.Dir.cwd().openFile(std.Io.Threaded.global_single_threaded.io(), path, .{ .mode = .read_only }) catch return error.SkipZigTest;
    defer file.close(std.Io.Threaded.global_single_threaded.io());
    const st = try file.stat(std.Io.Threaded.global_single_threaded.io());
    const data = try std.heap.page_allocator.alloc(u8, @intCast(st.size));
    defer std.heap.page_allocator.free(data);
    _ = try file.readPositionalAll(std.Io.Threaded.global_single_threaded.io(), data, 0);
    // attn_q blk.0 @ off=672048288 dims=[2560,2560] (verificado)
    const D: usize = 8351360;
    const off: usize = 672048288;
    const ne0: usize = 2560;
    const ne1: usize = 2560;
    const datab = (ne0 * ne1 * 2 + 7) / 8; // 1638400
    // Leemos 4 grupos (512 valores) del FINAL del tensor donde la escala
    // real SÍ está adyacente (offset datab = 1638400). Esto verifica
    // dequantI2_S con su layout "tensor parcial con su escala".
    const tail_groups = 4;
    const tail_n = tail_groups * 128; // 512
    const tail_data_bytes = (tail_n * 2 + 7) / 8; // 128
    const slice_start = datab - tail_data_bytes; // 1638272
    const slice = data[D + off + slice_start ..][0 .. tail_data_bytes + 4];
    const scale = @as(f32, @bitCast(std.mem.readInt(u32, slice[tail_data_bytes..][0..4], .little)));
    try std.testing.expect(scale > 0.5 and scale < 3.0);
    var out: [tail_n]f32 = undefined;
    ggufr.dequantI2_S(slice, &out);
    var n_neg: usize = 0;
    var n_zero: usize = 0;
    var n_pos: usize = 0;
    for (out) |v| {
        try std.testing.expect(v <= scale and v >= -scale);
        if (v < -scale * 0.9) n_neg += 1 else if (v > scale * 0.9) n_pos += 1 else n_zero += 1;
    }
    // Fracciones ternarias del modelo entrenado BitNet b1.58 (ground truth
    // python sobre el tensor completo): neg 25.2%, zero 49.6%, pos 25.2% —
    // NO ~50/50 como supuesto inicial: la mayoría de pesos son 0.
    const totalv: f32 = @floatFromInt(out.len);
    try std.testing.expect(@as(f32, @floatFromInt(n_neg)) / totalv > 0.15);
    try std.testing.expect(@as(f32, @floatFromInt(n_pos)) / totalv > 0.15);
    try std.testing.expect(@as(f32, @floatFromInt(n_zero)) / totalv > 0.30);
}

// lane-kvc P2+P3: carga E2E del GGUF BitNet real — arch bitnet-b1.58 aceptado
// por ModelConfig, QuantWeight i2_s dequantiza con la escala global del tail.
// Ground truth python: blk.0.attn_q scale=1.2188, ternario 25/50/25.
test "bitnet gguf: ModelConfig arch + QuantWeight i2_s E2E" {
    const t = std.testing;
    const path = "/ai/models/bitnet/ggml-model-i2_s.gguf";
    const io = std.Io.Threaded.global_single_threaded.io();
    var g = ggufr.GgufFile.fromFileMmap(io, t.allocator, path) catch return error.SkipZigTest;
    defer g.deinit();

    // P2: el arch del GGUF real es "bitnet-b1.58" — debe pasar el gate.
    const arch = g.arch() orelse return error.TestUnexpectedResult;
    try t.expect(model_config.ModelConfig.isSupportedArch(arch));
    try t.expect(model_config.ModelConfig.isBitnet(arch));

    const cfg = try model_config.ModelConfig.fromGguf(&g);
    try t.expectEqual(@as(usize, 30), cfg.block_count);
    try t.expectEqual(@as(usize, 2560), cfg.embedding_length);
    try t.expectEqual(@as(usize, 6912), cfg.feed_forward_length);
    try t.expectEqual(@as(usize, 20), cfg.head_count);
    try t.expectEqual(@as(usize, 5), cfg.head_count_kv); // GQA 4:1
    try t.expectEqual(@as(usize, 128256), cfg.vocab_size);
    try t.expectEqual(@as(usize, 128), cfg.rope_dimension_count);

    // P3: QuantWeight i2_s sobre blk.0.attn_q.weight — dequant + stats
    const qinfo = g.getTensor("blk.0.attn_q.weight") orelse return error.TestUnexpectedResult;
    try t.expectEqual(ggufr.GgmlType.i2_s, qinfo.dtype);
    const qw = qw_mod.QuantWeight.init(qinfo, g.tensorData(qinfo));
    try t.expect(qw.dtype() == .i2_s);

    // Subtensor rápido (rows [0,4), cols [0,8)) y stats del dequant completo
    // vía escala global: el rango de valores debe ser {-s, 0, +s} con
    // s = escala leída del tail (ground truth: 1.2188547849655151).
    const scale = @as(f32, @bitCast(std.mem.readInt(
        u32,
        g.tensorData(qinfo)[(@as(usize, @intCast(qinfo.numel())) * 2 + 7) / 8 ..][0..4],
        .little,
    )));
    try t.expectApproxEqRel(@as(f32, 1.2188547849655151), scale, 1e-5);

    var sub = try qw.get_subtensor(t.allocator, 0, 4, 0, 8);
    defer sub.deinit();
    // Ground truth python (out-rows 0-3, cols 0-7): valores exactamente en
    // {0, +s, -s} tras el roundtrip f16 (s±1e-3 rel por la conversión f16).
    for (sub.data) |v| {
        const vf: f32 = v;
        try t.expect(vf == 0.0 or
            @abs(vf - scale) < 1e-3 or
            @abs(vf + scale) < 1e-3);
    }
    // Valor puntual ground truth (layout del cuantizador de referencia,
    // shift 6-2*(j/32)): out-row 0 = [0, 0, 0, -s, 0, -s, 0, 0]
    try t.expectEqual(@as(f32, 0.0), sub.data[0]);
    try t.expectEqual(@as(f32, 0.0), sub.data[1]);
    try t.expectEqual(@as(f32, 0.0), sub.data[2]);
    try t.expectApproxEqRel(-scale, sub.data[3], 1e-3);
    try t.expectApproxEqRel(-scale, sub.data[5], 1e-3);
    // Out-row 3 (flat 7680+ ⇒ grupos 60-63 ≥ 32: ejercita el INTERLEAVING
    // del cuantizador — byte gp del bloque j del grupo-32 i en raw[i*1024
    // + j*32 + gp]). GT python: [0, -s, -s, 0, 0, -s, -s, -s].
    const r3 = sub.data[24..32];
    try t.expectEqual(@as(f32, 0.0), r3[0]);
    try t.expectApproxEqRel(-scale, r3[1], 1e-3);
    try t.expectApproxEqRel(-scale, r3[2], 1e-3);
    try t.expectEqual(@as(f32, 0.0), r3[3]);
    try t.expectEqual(@as(f32, 0.0), r3[4]);
    try t.expectApproxEqRel(-scale, r3[5], 1e-3);
    try t.expectApproxEqRel(-scale, r3[6], 1e-3);
    try t.expectApproxEqRel(-scale, r3[7], 1e-3);
}
