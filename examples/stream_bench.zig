//! stream_bench — harness standalone de streaming denso (Lane D, D4/D5).
//!
//! Driver directo de los pesos FFN del GGUF (la parte que domina el tráfico
//! por token): mide tok/s y bytes H2D/token con DOS pipelines sobre el mismo
//! modelo:
//!   wire (default)  — quantized-on-the-wire: bytes crudos QuantWeight suben
//!                     una vez por ventana de residencia (quantResidentPtr);
//!                     GEMM vía qgemmKernel para dtypes soportados.
//!   --noqwire       — pipeline VIEJO: dequant f32 a scratch + weight_cache
//!                     (8× bytes en Q4/Q8; la línea base a batir).
//!
//! Streaming: límite opcional de capas residentes (ZIG_AI_MAX_RESIDENT),
//! expulsión LRU por capa, upload lazy dentro del barrido por token (patrón
//! AirLLM). El overlap copy↔compute real llega en D5; aquí se valida la
//! contabilidad de bytes y el techo secuencial.
//!
//! Métrica D4: bytes H2D/token = Σ pesos CUANTIZADOS (no f32).
const std = @import("std");
const time = @import("time");
const debug = @import("debug");
const cudaz = @import("cudaz");
const gguf = @import("gguf");
const model_config = @import("model_config");
const quant_weight = @import("quant_weight");
const core = @import("core");
const Tensor = core.Tensor;
const matmul = @import("matmul");
const cublas = @import("cublas");
const layer_kernels = @import("layer_kernels");
const ext_mem = @import("cudaz_ext_mem");

/// qtype del kernel qgemmKernel para un dtype GGUF (tipos 0..7, lane-a/B).
fn qtypeOf(t: gguf.GgmlType) ?u32 {
    return switch (t) {
        .q4_0 => 0,
        .q4_1 => 1,
        .q5_k => 2,
        .q6_k => 3,
        .q4_k => 4,
        .q8_0 => 5,
        .q2_k => 7,
        .q3_k => 6,
        else => null,
    };
}

const LayerWeights = struct {
    gate: quant_weight.QuantWeight,
    up: quant_weight.QuantWeight,
    down: quant_weight.QuantWeight,

    fn totalQuantBytes(self: *const LayerWeights) usize {
        return self.gate.bytes.len + self.up.bytes.len + self.down.bytes.len;
    }
};

/// Estado de residencia de UNA capa en el pipeline del harness.
const LayerState = struct {
    resident: bool = false,
    last_used: u64 = 0,
    /// bit i (0=gate,1=up,2=down): 1 = ese peso va quantized-on-the-wire.
    wire_mask: u3 = 0,
    /// wire: puntero DEVICE (para lanzar qgemm) y clave HOST del mapa
    /// (@intFromPtr(w.bytes.ptr), para expulsar). ¡NO son intercambiables!
    /// Root-cause del known-issue: pasar el dev ptr a evictQuantCachePtr
    /// (que busca clave host) hacía fetchRemove fallar en silencio y las
    /// entradas sobrevivían a cada expulsión ⇒ hits eternos sin DMA.
    dev_ptrs: [3]usize = .{ 0, 0, 0 },
    host_keys: [3]usize = .{ 0, 0, 0 },
    /// fallback f32 por tensor: scratch transpuesto [out,in] vivo mientras
    /// residente + clave host del weight_cache (= data.ptr del scratch).
    scratches: [3][]f32 = .{ &.{}, &.{}, &.{} },
};

pub fn main(init: std.process.Init) !void {
    debug.init();
    const io = init.io;
    const gpa = init.gpa;

    var stdout_buffer: [0x2000]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    var model_path: ?[]const u8 = null;
    var tokens: usize = 6;
    var noqwire = false;
    var overlap = false;
    var args_it = std.process.Args.Iterator.init(init.minimal.args);
    _ = args_it.next();
    while (args_it.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--model=")) {
            model_path = arg["--model=".len..];
        } else if (std.mem.startsWith(u8, arg, "--tokens=")) {
            tokens = try std.fmt.parseInt(usize, arg["--tokens=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--noqwire")) {
            noqwire = true;
        } else if (std.mem.eql(u8, arg, "--overlap")) {
            overlap = true;
        }
    }
    const path = model_path orelse init.environ_map.get("GGUF_MODEL_PATH") orelse {
        try stdout.print("uso: stream_bench --model=RUTA.gguf [--tokens=N] [--noqwire]\n", .{});
        try stdout.flush();
        return error.MissingModel;
    };

    try stdout.print("== stream_bench (Lane D) — {s} ==\n", .{std.fs.path.basename(path)});
    try stdout.print("modo: {s}\n", .{if (noqwire) "f32-wire (VIEJO)" else "quantized-on-the-wire (D4)"});
    try stdout.flush();

    // ── GPU + engines ──
    if (!cudaz.isCudaAvailable()) return error.CudaUnavailable;
    try cudaz.ensureContext();
    // Sesión GPU exclusiva (disciplina .bench.lock). Ruta: env
    // ZIG_AI_BENCH_LOCK (mismo override que scripts/zig-ai-run) → default
    // raíz del repo (resuelta en runtime, sin paths absolutos).
    const bench_lock_path = blk: {
        if (std.c.getenv("ZIG_AI_BENCH_LOCK")) |v| break :blk std.mem.span(v);
        break :blk ".bench.lock"; // cwd = raíz del repo en el flujo normal
    };
    const lock_file = std.Io.Dir.cwd().createFile(io, bench_lock_path, .{ .truncate = false }) catch null;
    var have_lock = false;
    if (lock_file) |lf| {
        if (std.c.flock(lf.handle, 2 | 4) == 0) have_lock = true else lf.close(io); // EX|NB
    }
    if (!have_lock) {
        debug.dbg.printLevel(.info, "[stream_bench] GPU ocupada (.bench.lock), sigo sin lock\n", .{});
    }

    var engine = try matmul.MatmulEngine.init(gpa, .cublas, .f32);
    defer engine.deinit();
    const shared_raw = try matmul.MatmulEngine.sharedCudaStreamRaw();
    var lk = try layer_kernels.LayerKernels.init(shared_raw);

    // ── Modelo ──
    var g = try gguf.GgufFile.fromFileMmap(io, gpa, path);
    defer g.deinit();
    const cfg = try model_config.ModelConfig.fromGguf(&g);
    const emb = cfg.embedding_length;
    const inter = cfg.feed_forward_length;
    const nl = cfg.block_count;
    const max_resident: usize = if (init.environ_map.get("ZIG_AI_MAX_RESIDENT")) |mr|
        try std.fmt.parseInt(usize, mr, 10)
    else
        nl;

    try stdout.print("capas={d} n_embd={d} inter={d} | max_resident={d} noqwire={}\n", .{ nl, emb, inter, max_resident, noqwire });

    const g_ptr: *const gguf.GgufFile = &g;
    const layers = try gpa.alloc(LayerWeights, nl);
    for (layers, 0..) |*lw, i| {
        lw.gate = try loadW(g_ptr, i, "ffn_gate.weight");
        lw.up = try loadW(g_ptr, i, "ffn_up.weight");
        lw.down = try loadW(g_ptr, i, "ffn_down.weight");
    }
    const states = try gpa.alloc(LayerState, nl);
    for (states) |*s| s.* = .{};
    // Limpieza de scratches f32 del fallback al salir (los raw se expulsan vía
    // evictQuantCachePtr en el defer global del cache estático).
    defer for (states) |*s| {
        for (s.scratches) |sc| {
            if (sc.len > 0 and s.resident) gpa.free(sc);
        }
    };
    defer gpa.free(states);
    defer gpa.free(layers);

    // Activaciones device (bs=1 decode).
    const x_dev = try cublas.GpuTensor(f32).alloc(emb);
    defer x_dev.deinit();
    var gate_dev = try cublas.GpuTensor(f32).alloc(inter);
    defer gate_dev.deinit();
    var up_dev = try cublas.GpuTensor(f32).alloc(inter);
    defer up_dev.deinit();
    var h_dev = try cublas.GpuTensor(f32).alloc(inter);
    defer h_dev.deinit();
    var y_dev = try cublas.GpuTensor(f32).alloc(emb);
    defer y_dev.deinit();

    {
        const hx = try gpa.alloc(f32, emb);
        defer gpa.free(hx);
        var rng = std.Random.Xoshiro256.init(42);
        for (hx) |*v| v.* = rng.random().float(f32) - 0.5;
        try cudaz.cuMemcpyHtoD(x_dev.ptr(), @intFromPtr(hx.ptr), emb * @sizeOf(f32));
    }

    // ── D5: doble búfer copy↔compute (solo all-wire) ──
    if (overlap) return runOverlap(gpa, stdout, layers, x_dev.ptr(), emb, inter, tokens);

    // ── Bucle decode streaming ──
    var tick: u64 = 0;

    const t_all = time.Timer.start();
    matmul.MatmulEngine.quantDmaBytesReset();
    const dma_start = matmul.MatmulEngine.quantDmaBytes();

    for (0..tokens) |tok| {
        for (0..nl) |li| {
            const s = &states[li];
            const lw = &layers[li];
            tick += 1;

            if (!s.resident) {
                while (countResident(states) >= max_resident) {
                    try evictLru(states, gpa, &engine);
                }

                var moved: usize = 0;
                const weights = [3]*const quant_weight.QuantWeight{ &lw.gate, &lw.up, &lw.down };
                const dims = [3][2]usize{ .{ inter, emb }, .{ inter, emb }, .{ emb, inter } };
                inline for (0..3) |wi| {
                    const w = weights[wi];
                    if (!noqwire and qtypeOf(w.dtype()) != null) {
                        s.wire_mask |= @as(u3, 1) << wi;
                        s.host_keys[wi] = @intFromPtr(w.bytes.ptr); // clave del mapa
                        s.dev_ptrs[wi] = try matmul.MatmulEngine.quantResidentPtr(gpa, w.bytes);
                    } else {
                        const d0 = dims[wi][0];
                        const d1 = dims[wi][1];
                        s.scratches[wi] = try gpa.alloc(f32, d0 * d1);
                        w.dequantToF32Transposed(s.scratches[wi]);
                        try f32Upload(&engine, s.scratches[wi], d0, d1);
                        moved += s.scratches[wi].len * @sizeOf(f32);
                    }
                }
                s.resident = true;
                _ = &moved;
                if (debug.dbg.at(.detail)) {
                    debug.dbg.printLevel(.detail, "[stream_bench] tok={d} capa={d} subida {d:.1} MB (wire_mask={b})\n", .{ tok, li, @as(f64, @floatFromInt(moved)) / (1024.0 * 1024.0), s.wire_mask });
                }
            }
            s.last_used = tick;

            // Forward FFN bs=1: x→gate/up→swiglu→down→y ; y pasa a ser x.
            // Gating POR TENSOR (lo mismo que tendrá que hacer hybrid_layer/E):
            // cada proyección usa su ruta según dtype y disponibilidad kernel.
            {
                const shp_g = [_]usize{ inter, emb };
                const str_e = [_]usize{ emb, 1 };
                try proj(&lk, &engine, x_dev, &gate_dev, 1, emb, inter, (s.wire_mask >> 0) & 1 == 1, s.dev_ptrs[0], s.scratches[0], shp_g, str_e, qtypeOf(layers[li].gate.dtype()));
                try proj(&lk, &engine, x_dev, &up_dev, 1, emb, inter, (s.wire_mask >> 1) & 1 == 1, s.dev_ptrs[1], s.scratches[1], shp_g, str_e, qtypeOf(layers[li].up.dtype()));
                try lk.swiglu(gate_dev.ptr(), up_dev.ptr(), inter);
                const shp_d = [_]usize{ emb, inter };
                const str_i = [_]usize{ inter, 1 };
                try proj(&lk, &engine, h_dev, &y_dev, 1, inter, emb, (s.wire_mask >> 2) & 1 == 1, s.dev_ptrs[2], s.scratches[2], shp_d, str_i, qtypeOf(layers[li].down.dtype()));
            }
            // carry: y → x (DtoD; sin residual real — irrelevante para throughput)
            try cudaz.cuMemcpyDtoD(x_dev.ptr(), y_dev.ptr(), emb * @sizeOf(f32));
        }

        if (debug.dbg.at(.info)) {
            const ms = @as(f64, @floatFromInt(t_all.read())) / 1e6 / @as(f64, @floatFromInt(tok + 1));
            debug.dbg.printLevel(.info, "[stream_bench] token {d}: ms/token acumulado={d:.1}\n", .{ tok, ms });
        }
    }

    const wall_s = @as(f64, @floatFromInt(t_all.read())) / 1e9;
    const toks_f = @as(f64, @floatFromInt(tokens));
    // !! KNOWN-ISSUE (Lane D): bajo ZIG_AI_MAX_RESIDENT < nl los hits/misses
    // de este modo secuencial no cuadran con las rotaciones esperadas entre
    // tokens (ver HANDOFFS 12:4x). El modo --overlap no usa este código.
    const total_wire_bytes = matmul.MatmulEngine.quantDmaBytes() - dma_start;
    var qsum: usize = 0;
    var f32_equiv: usize = 0;
    for (layers) |*lw| {
        qsum += lw.totalQuantBytes();
        f32_equiv += (2 * inter * emb + emb * inter) * @sizeOf(f32);
    }

    try stdout.print("\n--- Resultados ---\n", .{});
    try stdout.print("tok/s = {d:.3}  ({d:.1} ms/token, {d} tokens)\n", .{ toks_f / wall_s, wall_s * 1000.0 / toks_f, tokens });
    try stdout.print("bytes H2D totales = {d:.2} MB ({d:.2} MB/token)\n", .{
        @as(f64, @floatFromInt(total_wire_bytes)) / (1024.0 * 1024.0),
        @as(f64, @floatFromInt(total_wire_bytes)) / (1024.0 * 1024.0) / toks_f,
    });
    try stdout.print("Σ FFN cuantizado (una pasada) = {d:.2} MB ; equivalente f32 = {d:.2} MB\n", .{
        @as(f64, @floatFromInt(qsum)) / (1024.0 * 1024.0),
        @as(f64, @floatFromInt(f32_equiv)) / (1024.0 * 1024.0),
    });
    try stdout.print("device cuantizado residente al final = {d:.1} MB\n", .{@as(f64, @floatFromInt(matmul.MatmulEngine.quantCacheBytes())) / (1024.0 * 1024.0)});

    // ── Proyección E2E honesta (copy-bound) ──
    // El harness mide SOLO FFN; el modelo completo streamea además atención,
    // SSM/ShortConv y normas. Bajo la hipótesis copy-bound (válida cuando
    // ms-copy ≥ ms-compute — cierto en 0.8B y en el 27B medido), el tiempo
    // por token escala con los BYTES: tok/s_e2e ≈ tok/s_ffn × ffn/total.
    var blk_bytes_total: usize = 0;
    var cls_ffn: usize = 0;
    var cls_attn: usize = 0;
    var cls_ssm: usize = 0;
    var cls_other: usize = 0;
    var tit = g.tensors.iterator();
    while (tit.next()) |kv| {
        const name = kv.key_ptr.*;
        if (!std.mem.startsWith(u8, name, "blk.")) continue;
        const b = kv.value_ptr.dataBytes();
        blk_bytes_total += b;
        // Patrones verificados contra GGUF reales (qwen3.5: attn_qkv/
        // attn_gate/ssm_a/ssm_conv1d sin puntos internos; lfm2 añade
        // .mlp_/.conv). Los *_norm caen en "otros" a propósito.
        const has_ffn = std.mem.indexOf(u8, name, ".ffn_") != null;
        const has_attn = std.mem.indexOf(u8, name, ".attn_") != null or std.mem.indexOf(u8, name, ".self_attn.") != null;
        const has_ssm = std.mem.indexOf(u8, name, ".ssm") != null or std.mem.indexOf(u8, name, ".shortconv.") != null or std.mem.indexOf(u8, name, ".conv.") != null;
        if (has_ffn) {
            cls_ffn += b;
        } else if (has_attn) {
            cls_attn += b;
        } else if (has_ssm) {
            cls_ssm += b;
        } else {
            cls_other += b;
        }
    }
    if (qsum > 0 and blk_bytes_total >= qsum) {
        const ratio = @as(f64, @floatFromInt(qsum)) / @as(f64, @floatFromInt(blk_bytes_total));
        const toks_e2e = (toks_f / wall_s) * ratio;
        try stdout.print("proyección E2E copy-bound: Σ blk.* = {d:.2} MB/pasada ⇒ ~{d:.3} tok/s ({d:.1}% del tráfico es FFN)\n", .{
            @as(f64, @floatFromInt(blk_bytes_total)) / (1024.0 * 1024.0),
            toks_e2e,
            ratio * 100.0,
        });
        try stdout.print("desglose por clase: ffn={d:.1} attn={d:.1} ssm/conv={d:.1} otros(norms)={d:.1} MB\n", .{
            @as(f64, @floatFromInt(cls_ffn)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(cls_attn)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(cls_ssm)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(cls_other)) / (1024.0 * 1024.0),
        });
        try stdout.print("(no incluye embedding/lm_head residentes ni cómputo no-oculto; techo duro = PCIe)\n", .{});
    }
    try stdout.flush();

    if (lock_file) |lf| {
        _ = std.c.flock(lf.handle, 8);
        lf.close(io);
    }
}

/// Coreografía D5 (estilo FreeToken offload_cache.py:602-798) sobre slots
/// rotatorios [layer%2]:
///   - fence inicial: la primera copia espera TODO lo encolado en compute
///     (ev_begin), para no pisar pesos que un kernel vivo está leyendo.
///   - prefetch N+1 se ENCOLA en copyStream justo tras encolar compute N:
///     copy-engine y SM trabajan a la vez (solape real, sin sync de host).
///   - ev_ready[slot]: compute espera a que la copia del slot termine.
///   - ev_free[slot]: grabado en compute tras el ÚLTIMO kernel que lee el
///     slot; la siguiente copia a ese slot espera antes de machacar bytes.
///   - víctimas: con rotación %2 el slot víctima ES el otro búfer (equivalente
///     a marcar usage=0 las entradas viejas).
fn runOverlap(
    gpa: std.mem.Allocator,
    stdout: anytype,
    layers: []LayerWeights,
    x_dev_ptr: usize,
    emb: usize,
    inter: usize,
    tokens: usize,
) !void {
    const nl = layers.len;
    // All-wire obligatorio para este modo.
    for (layers) |*lw| {
        if (qtypeOf(lw.gate.dtype()) == null or qtypeOf(lw.up.dtype()) == null or qtypeOf(lw.down.dtype()) == null)
            return error.OverlapRequiresAllWire;
    }

    try stdout.print("\nmodo: OVERLAP copy↔compute (D5, 2 slots)\n", .{});
    try stdout.flush();

    // Slab pinned único con todos los bytes cuantizados (fuente async válida;
    // pin-after-fill: primero copiamos mmap→slab, luego registramos).
    var offsets = try gpa.alloc(usize, nl * 3);
    defer gpa.free(offsets);
    var total: usize = 0;
    for (layers, 0..) |*lw, i| {
        const ws = [3][]const u8{ lw.gate.bytes, lw.up.bytes, lw.down.bytes };
        for (ws, 0..) |b, wi| {
            offsets[i * 3 + wi] = total;
            total += b.len;
        }
    }
    const slab_raw = try cudaz.cuMemAllocHost(total);
    defer cudaz.cuMemFreeHost(slab_raw);
    const slab: [*]u8 = @ptrCast(slab_raw);
    {
        // fill una vez (pin-after-fill: tocar ANTES de registrar sería
        // redundante aquí porque cuMemAllocHost ya devuelve pinned; el patrón
        // HostBank aplica cuando la fuente es un mmap externo).
        for (layers, 0..) |*lw, i| {
            const ws = [3][]const u8{ lw.gate.bytes, lw.up.bytes, lw.down.bytes };
            for (ws, 0..) |b, wi| {
                @memcpy(slab[offsets[i * 3 + wi]..][0..b.len], b);
            }
        }
        // SIN hostRegister aquí: cuMemAllocHost YA es pinned (re-registrar da
        // ERROR_INVALID_VALUE benigno pero ruidoso — visto en 27B val 00:42).
        // El patrón HostBank aplica a mmaps externos, no a pin propio.
    }

    // Slots device: tamaños máximos por peso.
    var max_gu: usize = 0;
    var max_dn: usize = 0;
    for (layers) |*lw| {
        max_gu = @max(max_gu, lw.gate.bytes.len);
        max_gu = @max(max_gu, lw.up.bytes.len);
        max_dn = @max(max_dn, lw.down.bytes.len);
    }
    const Slot = struct {
        dev_gate: cudaz.CUdeviceptr,
        dev_up: cudaz.CUdeviceptr,
        dev_down: cudaz.CUdeviceptr,
        ready: cudaz.CUevent,
        free_ev: cudaz.CUevent,
        layer: ?usize = null,
    };
    var slots: [2]Slot = undefined;
    for (&slots) |*sl| {
        sl.* = .{
            .dev_gate = try cudaz.cuMemAlloc(max_gu),
            .dev_up = try cudaz.cuMemAlloc(max_gu),
            .dev_down = try cudaz.cuMemAlloc(max_dn),
            .ready = try cudaz.cuEventCreate(0),
            .free_ev = try cudaz.cuEventCreate(0),
        };
    }
    defer for (&slots) |*sl| {
        cudaz.cuMemFree(sl.dev_gate);
        cudaz.cuMemFree(sl.dev_up);
        cudaz.cuMemFree(sl.dev_down);
        cudaz.cuEventDestroy(sl.ready);
        cudaz.cuEventDestroy(sl.free_ev);
    };

    const compute_stream = try matmul.MatmulEngine.sharedCudaStreamRaw();
    const copy_stream = try matmul.MatmulEngine.copyStreamRaw();
    const ev_begin = try cudaz.cuEventCreate(0);
    defer cudaz.cuEventDestroy(ev_begin);
    const ev_tok_cstart = try cudaz.cuEventCreate(0);
    defer cudaz.cuEventDestroy(ev_tok_cstart);
    const ev_tok_cend = try cudaz.cuEventCreate(0);
    defer cudaz.cuEventDestroy(ev_tok_cend);
    const ev_tok_kstart = try cudaz.cuEventCreate(0);
    defer cudaz.cuEventDestroy(ev_tok_kstart);
    const ev_tok_kend = try cudaz.cuEventCreate(0);
    defer cudaz.cuEventDestroy(ev_tok_kend);

    var lk = try layer_kernels.LayerKernels.init(compute_stream);

    // Activaciones propias del modo overlap.
    const x_dev = try cublas.GpuTensor(f32).alloc(emb);
    defer x_dev.deinit();
    const gate_dev = try cublas.GpuTensor(f32).alloc(inter);
    defer gate_dev.deinit();
    const up_dev = try cublas.GpuTensor(f32).alloc(inter);
    defer up_dev.deinit();
    const h_dev = try cublas.GpuTensor(f32).alloc(inter);
    defer h_dev.deinit();
    const y_dev = try cublas.GpuTensor(f32).alloc(emb);
    defer y_dev.deinit();
    {
        const hx = try gpa.alloc(f32, emb);
        defer gpa.free(hx);
        var rng = std.Random.Xoshiro256.init(42);
        for (hx) |*v| v.* = rng.random().float(f32) - 0.5;
        try cudaz.cuMemcpyHtoD(x_dev.ptr(), @intFromPtr(hx.ptr), emb * @sizeOf(f32));
    }

    // fence inicial: la primera copia de ESTE token espera al compute previo.
    try cudaz.cuEventRecord(ev_begin, compute_stream);
    try ext_mem.streamWaitEvent(@ptrCast(copy_stream), @ptrCast(ev_begin), 0);

    var ms_copy_total: f64 = 0;
    var ms_compute_total: f64 = 0;
    var copy_bytes_total: usize = 0;

    const t_all = time.Timer.start();

    for (0..tokens) |_| {
        // Marcas de tiempo del token (GPU-side, sin sincronizar el medio).
        _ = try cudaz.cuEventRecord(ev_tok_cstart, copy_stream);
        _ = try cudaz.cuEventRecord(ev_tok_kstart, compute_stream);
        for (0..nl) |li| {
            const slot_i = li % 2;
            const sl = &slots[slot_i];
            const lw = &layers[li];

            if (sl.layer != li) {
                // El búfer puede estar siendo leído por kernels encolados
                // previamente (capa li-2): esperar su release.
                if (sl.layer != null) {
                    try ext_mem.streamWaitEvent(@ptrCast(copy_stream), @ptrCast(sl.free_ev), 0);
                }
                try cudaz.cuMemcpyHtoDAsync(sl.dev_gate, @intFromPtr(slab + offsets[li * 3 + 0]), lw.gate.bytes.len, copy_stream);
                try cudaz.cuMemcpyHtoDAsync(sl.dev_up, @intFromPtr(slab + offsets[li * 3 + 1]), lw.up.bytes.len, copy_stream);
                try cudaz.cuMemcpyHtoDAsync(sl.dev_down, @intFromPtr(slab + offsets[li * 3 + 2]), lw.down.bytes.len, copy_stream);
                copy_bytes_total += lw.totalQuantBytes();
                try cudaz.cuEventRecord(sl.ready, copy_stream);
                sl.layer = li;
            }

            // compute espera la copia de SU slot (cross-stream).
            try ext_mem.streamWaitEvent(@ptrCast(compute_stream), @ptrCast(sl.ready), 0);

            // Kernels de compute (encolados; se ejecutan solapados con copias).
            try lk.qgemm(x_dev.ptr(), sl.dev_gate, gate_dev.ptr(), 1, emb, inter, qt_g_of(layers[li], .gate));
            try lk.qgemm(x_dev.ptr(), sl.dev_up, up_dev.ptr(), 1, emb, inter, qt_g_of(layers[li], .up));
            try lk.swiglu(gate_dev.ptr(), up_dev.ptr(), inter);
            try lk.qgemm(h_dev.ptr(), sl.dev_down, y_dev.ptr(), 1, inter, emb, qt_g_of(layers[li], .down));
            try cudaz.cuMemcpyDtoD(x_dev_ptr, y_dev.ptr(), emb * @sizeOf(f32));

            // Release: último uso de este slot por AHORA.
            try cudaz.cuEventRecord(sl.free_ev, compute_stream);
        }

        _ = try cudaz.cuEventRecord(ev_tok_cend, copy_stream);
        _ = try cudaz.cuEventRecord(ev_tok_kend, compute_stream);

        // Stats por token: un solo par de eventos por stream, un solo sync.
        try cudaz.cuEventSynchronize(ev_tok_kend);
        try cudaz.cuEventSynchronize(ev_tok_cend);
        var cms: f32 = 0;
        try cudaz.cuEventElapsedTime(&cms, ev_tok_cstart, ev_tok_cend);
        ms_copy_total += cms;
        var kms: f32 = 0;
        try cudaz.cuEventElapsedTime(&kms, ev_tok_kstart, ev_tok_kend);
        ms_compute_total += kms;
    }

    const wall_s = @as(f64, @floatFromInt(t_all.read())) / 1e9;
    try stdout.print("--- Overlap (D5) ---\n", .{});
    try stdout.print("tok/s = {d:.3}  ({d:.1} ms/token, {d} tokens)\n", .{ @as(f64, @floatFromInt(tokens)) / wall_s, wall_s * 1000.0 / @as(f64, @floatFromInt(tokens)), tokens });
    try stdout.print("ms-copy/token = {d:.2}  ms-compute/token = {d:.2}  ratio c/k = {d:.2}\n", .{
        ms_copy_total / @as(f64, @floatFromInt(tokens)),
        ms_compute_total / @as(f64, @floatFromInt(tokens)),
        ms_copy_total / @max(ms_compute_total, 0.001),
    });
    try stdout.print("bytes H2D/token = {d:.2} MB (Σ cuantizado)\n", .{@as(f64, @floatFromInt(copy_bytes_total)) / (1024.0 * 1024.0) / @as(f64, @floatFromInt(tokens))});
    try stdout.flush();
}

fn tensorOf(data: []f32, shape: []const usize, strides: []usize) Tensor(f32) {
    return .{ .data = data, .shape = @constCast(shape), .strides = strides, .offset = 0, .allocator = null, .owns_data = false };
}

/// Sube/cachea el scratch f32 (device ptr para lanzar) — la CLAVE de
/// expulsión es aparte: @intFromPtr(data.ptr) (ver evictLru).
fn f32Upload(engine: *matmul.MatmulEngine, data: []f32, d0: usize, d1: usize) !void {
    var shape = [_]usize{ d0, d1 };
    var strides = [_]usize{ d1, 1 };
    _ = try engine.projectionDevicePtr(tensorOf(data, &shape, &strides));
}

fn loadW(g: *const gguf.GgufFile, i: usize, name: []const u8) !quant_weight.QuantWeight {
    var buf: [80]u8 = undefined;
    const full = try std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ i, name });
    const info = g.getTensor(full) orelse return error.TensorNotFound;
    return quant_weight.QuantWeight.init(info, g.tensorData(info));
}

fn countResident(states: []LayerState) usize {
    var n: usize = 0;
    for (states) |*s| {
        if (s.resident) n += 1;
    }
    return n;
}

/// Expulsa la capa residente más vieja (device cache + scratches f32).
fn evictLru(states: []LayerState, gpa: std.mem.Allocator, engine: *matmul.MatmulEngine) !void {
    var victim: ?usize = null;
    var oldest: u64 = std.math.maxInt(u64);
    for (states, 0..) |*s, i| {
        if (s.resident and s.last_used < oldest) {
            oldest = s.last_used;
            victim = i;
        }
    }
    const vi = victim orelse return error.NoResidentLayer;
    const s = &states[vi];
    inline for (0..3) |wi| {
        if ((s.wire_mask >> wi) & 1 == 1) {
            // CLAVE HOST (mmap), no el dev ptr — ver LayerState.
            _ = matmul.MatmulEngine.evictQuantCachePtr(s.host_keys[wi]);
        } else {
            // weight_cache keyed por data.ptr del SCRATCH host.
            if (s.scratches[wi].len > 0) {
                _ = engine.evictWeightCachePtr(@intFromPtr(s.scratches[wi].ptr));
                gpa.free(s.scratches[wi]);
                s.scratches[wi] = &.{};
            }
        }
    }
    s.wire_mask = 0;
    s.resident = false;
}

const WhichW = enum { gate, up, down };
fn qt_g_of(lw: LayerWeights, w: WhichW) u32 {
    const t = switch (w) {
        .gate => lw.gate.dtype(),
        .up => lw.up.dtype(),
        .down => lw.down.dtype(),
    };
    return qtypeOf(t).?;
}

/// Una proyección FFN: ruta wire (qgemm sobre bytes crudos residentes) o
/// ruta f32 (weight_cache + cuBLAS), según el gating del caller.
fn proj(
    lk: *layer_kernels.LayerKernels,
    engine: *matmul.MatmulEngine,
    x: cublas.GpuTensor(f32),
    out: *cublas.GpuTensor(f32),
    m: usize,
    k: usize,
    n: usize,
    use_wire: bool,
    dev_key: usize,
    scratch: []f32,
    shape: [2]usize,
    strides: [2]usize,
    qt: ?u32,
) !void {
    if (use_wire and qt != null) {
        try lk.qgemm(x.ptr(), dev_key, out.ptr(), m, k, n, qt.?);
    } else {
        var shp = shape;
        var str = strides;
        try engine.linearProjectionDevice(x, tensorOf(scratch, &shp, &str), out, m, k, n);
    }
}
