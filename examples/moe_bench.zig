//! moe_bench — benchmark standalone del pipeline MoE offload GPU-only
//! (Lane E). Carga un GGUF MoE real vía el parser zero-copy, monta slot
//! cache LRU global + gatherer + capas MoE, y mide ms/paso del FFN MoE en
//! decode bs=1 con activaciones sintéticas (el executor CPU y la generación
//! greedy completa llegan por tickets de F/C).
//!
//! Uso:
//!   MOE_MODEL=/ruta/modelo.gguf zig build moe-bench -- <n_pasos>
//!   # breadcrumbs: MOE_DEBUG=1 · A/B gather: NOGATHER=1
//!
//! Sin MOE_MODEL (o si no es MoE) imprime las specs detectadas y termina 0.
const builtin = @import("builtin");
const std = @import("std");
const gguf = @import("gguf");
const gguf_moe = @import("gguf_moe");
const cudaz = @import("cudaz");
const layer_kernels = @import("layer_kernels");
const moe_cuda = @import("moe_cuda");
const offload_cache = @import("offload_cache");
const moe_layer = @import("moe_layer");
const moe_cpu_gemv = @import("moe_cpu_gemv");
const moe_cpu_executor = @import("moe_cpu_executor");
const host_bank = @import("host_bank");
const expert_bundle = @import("expert_bundle"); // lane-e 11.2: bundle contiguo

/// Modo skew (env MOE_SKEW=zipf): construye x cuya fila dominante del router
/// sigue una distribución Zipf-ish sobre expertos ⇒ routing con skew realista
/// para estresar el LRU (hallazgo FreeToken §7.4) sin modelo real.
fn makeBiasedX(
    rng: *std.Random.Xoshiro256,
    router_f32: []const f32,
    n_embd: usize,
    target: usize,
    out: []f32,
) void {
    const row = router_f32[target * n_embd ..][0..n_embd];
    var norm2: f32 = 0;
    for (row) |v| norm2 += v * v;
    const scale = @sqrt(norm2) + 1e-6;
    for (out, 0..) |*v, i| {
        v.* = row[i] + (rng.random().float(f32) - 0.5) * 0.1 * scale;
    }
}

/// Zipf-ish simple: k-ésimo con peso 1/(k+1), muestreo inverso barato.
fn zipfTarget(rand_rand: std.Random, n_expert: usize) usize {
    const u = rand_rand.float(f32);
    var denom: f32 = 0;
    for (0..n_expert) |k| denom += 1.0 / @as(f32, @floatFromInt(k + 1));
    const threshold = u * denom;
    var acc: f32 = 0;
    for (0..n_expert) |k| {
        acc += 1.0 / @as(f32, @floatFromInt(k + 1));
        if (acc >= threshold) return k;
    }
    return n_expert - 1;
}

/// Mapeo dtype de bancos → formato del GEMV CPU (lane-f). null = no soportado.
fn ggmlToCpuFormat(t: @import("gguf").GgmlType) ?moe_cpu_gemv.Format {
    return switch (t) {
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q5_k => .q5_k,
        .q6_k => .q6_k,
        .q8_0 => .q8_0,
        .q4_k => .q4_k,
        .q3_k => .q3_k,
        .q2_k => .q2_k,
        .iq3_s => .iq3_s,
        .iq2_s => .iq2_s,
        // 4.6: desbloquea gemma-4 IQ2_XXS como modelo A/B (down=iq4_nl,
        // gate/up=iq2_xxs + composeCanonical de escalas externas 4.4).
        .iq4_nl => .iq4_nl,
        .iq2_xxs => .iq2_xxs,
        else => null,
    };
}

pub fn main() !void {
    // Allocator: DebugAllocator thread-safe. NOTA (race conocido): con
    // c_allocator, el camino híbrido (submits/syncs desde main + workers
    // pineados) corrompe el heap glibc ("double free or corruption !prev")
    // tras ~64 pasos; con DebugAllocator(.safety=true) el camino completa
    // 8 capas × 64 pasos sin corrupción — la carrera la absorben los checks.
    // El executor es el sospechoso (freePartial/staging vs workers); ver
    // ticket en HANDOFFS. Perf del bench no se ve afectada de forma
    // significativa (el cómputo domina).
    var dbg_alloc = std.heap.DebugAllocator(.{ .safety = true, .thread_safe = true }){};
    const gpa = dbg_alloc.allocator();
    defer {
        const leaked = dbg_alloc.deinit();
        if (leaked == .ok) std.debug.print("[bench] heap OK\n", .{}) else std.debug.print("[bench] LEAK reported (page-align allocations del streamer)\n", .{});
    }
    const io = std.Io.Threaded.global_single_threaded.io();

    const model_path = std.c.getenv("MOE_MODEL") orelse {
        std.debug.print("uso: MOE_MODEL=<gguf MoE> moe-bench [n_pasos]\n", .{});
        return;
    };

    // Lock de sesión GPU (flock exclusivo bloqueante; EINTR-retry).
    // NOTA: al fallo no se hace close() del fd (el fileClose del stack Io
    // panea en este camino); el fd muere con el proceso, que es ahora mismo.
    {
        const lock_dir = std.Io.Dir.cwd();
        const lf = try lock_dir.createFile(io, ".bench.lock", .{ .truncate = false });
        var tries: u32 = 0;
        while (if (builtin.target.os.tag != .windows) std.c.flock(lf.handle, 2) != 0 else false) {
            tries += 1;
            if (tries >= 3) {
                std.debug.print("SKIP: .bench.lock ocupada (EINTR x{d})\n", .{tries});
                return;
            }
            var ts: std.c.timespec = .{ .sec = 1, .nsec = 0 };
            if (builtin.target.os.tag == .windows) {
                std.Thread.sleep(std.time.ns_per_s);
            } else {
                _ = std.c.nanosleep(&ts, null);
            }
        }
    }

    // 4.3' copy-once: una ÚNICA copia PINNED anónima del GGUF sustituye el
    // mmap — el registro DEVICEMAP del driver SÓLO acepta memoria anónima
    // (4.3 refutado con control). Con ZIG_AI_MOE_COPYONCE=1 el GgufFile se
    // parsea TOMANDO PRESTADO el buffer (fromBytesBorrowed): cero mmap ⇒
    // RAM = tamaño del fichero (no 2×). Las capas comparten el mismo buffer.
    var copyonce_bank: ?host_bank.HostBank = null;
    var g = blk: {
        if (std.c.getenv("ZIG_AI_MOE_COPYONCE") != null) {
            const co = try host_bank.HostBank.fromFileCopyOnce(gpa, io, std.mem.span(model_path));
            copyonce_bank = co.bank;
            break :blk try gguf.GgufFile.fromBytesBorrowed(gpa, co.buf);
        }
        break :blk try gguf.GgufFile.fromFileMmap(io, gpa, std.mem.span(model_path));
    };
    // Orden de frees: g (borrowed) ANTES del bank (el buffer es su backing).
    defer {
        if (copyonce_bank) |*b| {
            g.deinit();
            b.unreg();
            b.deinit();
        } else {
            g.deinit();
        }
    }

    if (!gguf_moe.isMoeModel(&g)) {
        std.debug.print("{s}: no es MoE (sin router)\n", .{model_path});
        return;
    }
    const info = try gguf_moe.moeInfo(&g);
    std.debug.print("MoE detectado: family={s} E={d} top_k={d}\n", .{ @tagName(info.family), info.n_expert, info.top_k });

    var n_moe_layers: usize = 0;
    var first_spec: ?gguf_moe.MoeLayerSpec = null;
    const bc = gguf_moe.blockCountMeta(&g, g.arch().?) orelse 0;
    for (0..@intCast(bc)) |il| {
        if (!gguf_moe.isMoeLayer(&g, il)) continue;
        n_moe_layers += 1;
        if (first_spec == null) {
            const s = gguf_moe.layerSpec(&g, il) catch |err| {
                std.debug.print("capa {d}: spec inválida ({s})\n", .{ il, @errorName(err) });
                continue;
            };
            first_spec = s;
        }
    }
    std.debug.print("capas MoE: {d}/{d}\n", .{ n_moe_layers, bc });
    const spec = first_spec orelse return;

    // ── Pipeline GPU ──
    try cudaz.ensureContext();
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try layer_kernels.LayerKernels.init(stream);

    // Presupuesto v1: pool que cubre TODOS los expertos de UNA capa
    // (offload puro sin misses en steady state para una capa a la vez).
    const cfg = offload_cache.Config{
        .num_layers = @intCast(n_moe_layers),
        .num_experts = info.n_expert,
        // MOE_CACHE=N constriñe el pool (< n_expert) ⇒ churn/eviction medibles
        .cache_size = blk2: {
            if (std.c.getenv("MOE_CACHE")) |a| {
                if (std.fmt.parseInt(u32, std.mem.span(a), 10)) |n| break :blk2 @max(1, n) else |_| {}
            }
            break :blk2 info.n_expert;
        },
        .max_fetch = info.top_k * 2,
    };
    var cache_gpu = try moe_cuda.MoeCacheGpu.init(cfg);
    defer cache_gpu.deinit();
    var gatherer = try moe_cuda.ExpertGatherer.init(3);
    defer gatherer.deinit();

    // Contrato 5 real: registrar el mmap COMPLETO del GGUF (pin-after-fill)
    // ⇒ las fuentes del gather son VAs directas del archivo (cero copias).
    // A4: TODAS las capas MoE comparten cache_gpu+gatherer (pool único).
    var layers: std.ArrayList(moe_layer.MoeLayer) = .empty;
    defer {
        for (layers.items) |*l| l.deinit(gpa);
        layers.deinit(gpa);
    }
    // E2 (lane-e, 4.5-b fix): los Executors híbridos creados abajo (línea
    // ~274, initFull) NO se deinían en NINGÚN exit path — 2 executors ×
    // {struct, threads[], core_ids[]} = exactamente los 6 leaks que
    // reportaba el DebugAllocator al exit. Se manifiesta solo con
    // ZIG_AI_HYBRID=1 Y run limpio hasta el final (bajo contención VRAM
    // el CudaError mataba antes del reporte — de ahí lo "intermitente").
    // El deinit ANTES que las capas: los layers aún los referencian.
    var exec_gu: ?*moe_cpu_executor.Executor = null;
    var exec_dn: ?*moe_cpu_executor.Executor = null;
    defer {
        if (exec_gu) |e| e.deinit();
        if (exec_dn) |e| e.deinit();
    }
    const bc_all = gguf_moe.blockCountMeta(&g, g.arch().?) orelse 1;
    // 4.3': el region ES g.data — con copy-once es el buffer pinned
    // (fromBytesBorrowed), con mmap es la vista del archivo. Ambas rutas
    // terminan en el mismo gather (VAs directas).
    const region: []const u8 = g.data;
    // ── 11.2 p2 (lane-e): abrir el bundle si ZIG_AI_BUNDLE=<path> ──
    // Validación de layout contra el GGUF + homogeneidad (stride del
    // gather == slot del bundle) + mmap registrado pin-after-fill. El
    // bundle SOLO transporta bancos de expertos; specs/router siguen
    // del GGUF.
    var bundle_src: ?*expert_bundle.BundleSource = null;
    defer {
        if (bundle_src) |bs| {
            bs.deinit();
            gpa.destroy(bs);
        }
    }
    if (std.c.getenv("ZIG_AI_BUNDLE")) |bpath| blk: {
        if (copyonce_bank != null) {
            std.debug.print("[moe_bench] ZIG_AI_BUNDLE ignorado: COPYONCE activo\n", .{});
            break :blk;
        }
        const bs = try gpa.create(expert_bundle.BundleSource);
        errdefer gpa.destroy(bs);
        bs.* = expert_bundle.BundleSource.open(gpa, std.mem.span(bpath)) catch |e| {
            gpa.destroy(bs);
            std.debug.print("[moe_bench] bundle open falló ({s}) — fuentes: mmap GGUF\n", .{@errorName(e)});
            break :blk;
        };
        expert_bundle.validateAgainstGguf(bs, &g, gpa) catch |e| {
            bs.deinit();
            gpa.destroy(bs);
            std.debug.print("[moe_bench] bundle layout ≠ GGUF ({s}) — fuentes: mmap GGUF\n", .{@errorName(e)});
            break :blk;
        };
        // p2-v2: SIN registro (file-backed no registrable, driver 580 —
        // refutado 4.3). Ventanas pageable → staging clásico por capa.
        try bs.ensureMmap(io);
        bundle_src = bs;
        moe_layer.g_bundle_source = .{
            .base = bs.mmap.?.memory,
            .slot_bytes = .{ bs.hdr.slot_bytes[0], bs.hdr.slot_bytes[1], bs.hdr.slot_bytes[2] },
            .n_experts = bs.hdr.n_experts,
            .n_layers = bs.hdr.n_layers,
            .payload_offset = bs.hdr.payload_offset,
        };
        moe_layer.g_bundle_layer_seq = 0;
        std.debug.print("[moe_bench] bundle 11.2 p2-v2: {d} MB mapeados (pageable, staging clásico, CERO pinned), L={d} E={d} slot={d}B\n", .{
            (@as(usize, bs.hdr.slot_bytes[0]) + bs.hdr.slot_bytes[1] + bs.hdr.slot_bytes[2]) * bs.hdr.n_experts * bs.hdr.n_layers / 1_000_000,
            bs.hdr.n_layers,
            bs.hdr.n_experts,
            bs.hdr.slot_bytes[0],
        });
    }
    if (copyonce_bank != null) {
        // 4.4: componer escalas externas IN-PLACE (una pasada global) y
        // marcar el buffer como pre-registrado — las capas lo usan por VAs
        // directas sin copia alguna. El buffer pinned es mutable por
        // construcción (cuMemHostAlloc); region []const es solo la vista.
        const mut_region: []u8 = @constCast(region);
        const n_composed = gguf_moe.composeExternalScalesInPlace(&g, mut_region);
        moe_layer.g_pinned_preregistered = region;
        std.debug.print("[moe_bench] copy-once: {d} MB pinned anónimo + {d} bancos con escalas externas compuestos in-place\n", .{ region.len / 1_000_000, n_composed });
    }
    for (0..@intCast(bc_all)) |il| {
        if (!gguf_moe.isMoeLayer(&g, il)) continue;
        const spec_i = try gguf_moe.layerSpec(&g, il);
        const ml = try moe_layer.MoeLayer.init(gpa, spec_i, cfg, stream, &lk, &cache_gpu, &gatherer, region);
        // 11.2 p2-v2 (lane-e): las fuentes vienen del g_bundle_source
        // (ventanas pageable del bundle dentro de init — CERO pinned).
        try layers.append(gpa, ml);
    }
    if (layers.items.len == 0) return;
    // El global bundle solo aplica al attach de ARRIBA: reset inmediato
    // (otro caller posterior de MoeLayer.init no debe heredar las ventanas).
    moe_layer.g_bundle_source = null;

    // ── Lane F: attach executor CPU si ZIG_AI_HYBRID=1 (Contrato 8) ──
    // Geometrías desde el spec real del modelo; dtypes no cubiertos por el
    // GEMV CPU ⇒ offload puro con aviso (salvaguarda de moe_layer).
    // ── Lane F sonda aislada del executor sobre GPU real (ZIG_AI_PROBE_EXEC=1) ──
    if (std.c.getenv("ZIG_AI_PROBE_EXEC") != null) {
        const cgf2 = moe_cpu_gemv;
        const f_t: cgf2.Format = .q4_1;
        const ex = moe_cpu_executor.Executor.initFull(gpa, .{ .fmt = f_t, .k_dim = spec.gate.in_dim, .out_dim = spec.gate.out_dim, .n_experts = spec.gate.n_expert }) catch |err| {
            std.debug.print("[probe] init err {s}\n", .{@errorName(err)});
            return;
        };
        defer ex.deinit();
        std.debug.print("[probe] 1 init OK\n", .{});
        _ = ex.attachStream(@intFromPtr(stream));
        std.debug.print("[probe] 2 attach OK modo={s}\n", .{@tagName(ex.mode)});
        // Contrato serializado/hostfunc: hidden_dev es puntero HOST.
        const k_in: usize = spec.gate.in_dim;
        const hx = try gpa.alloc(f32, k_in);
        defer gpa.free(hx);
        for (hx) |*v| v.* = 0.25;
        std.debug.print("[probe] 3 alloc OK\n", .{});
        std.debug.print("[probe] 4 host-buf OK\n", .{});
        // pesos: usar el banco real del spec (host VAs del mmap)
        var ids_cpu = [_]i32{ 0, 1 };
        _ = &ids_cpu;
        const wq: []const f32 = @as([*]const f32, @ptrCast(@alignCast(spec.gate.bytes.ptr)))[0 .. spec.gate.bytes.len / 4];
        if (moe_cpu_executor.hostAllocPub(2048)) |b| {
            std.debug.print("[probe] 5a hostAlloc 2K OK\n", .{});
            moe_cpu_executor.hostFreePub(@alignCast(b));
        } else std.debug.print("[probe] 5a hostAlloc FAIL\n", .{});
        std.debug.print("[probe] 5 pre-submit\n", .{});
        const pend = ex.submit(99, @intFromPtr(hx.ptr), wq, ids_cpu[0..]) catch |err| {
            std.debug.print("[probe] submit err {s}\n", .{@errorName(err)});
            return;
        };
        const outp = ex.sync(pend) catch |err| {
            std.debug.print("[probe] sync err {s}\n", .{@errorName(err)});
            return;
        };
        std.debug.print("[probe] OK out[0]={d} out[1]={d} (parcial executor sobre GPU stream)\n", .{ outp[0], outp[1] });
        return; // la sonda termina aquí: no entra el pipeline completo
    }
    const hyb_on = blk2x: {
        const v = std.c.getenv("ZIG_AI_HYBRID") orelse break :blk2x false;
        break :blk2x std.mem.eql(u8, std.mem.span(v), "1");
    };
    if (hyb_on) {
        const f_gu: ?moe_cpu_gemv.Format = ggmlToCpuFormat(spec.gate.dtype);
        const f_dn: ?moe_cpu_gemv.Format = ggmlToCpuFormat(spec.down.dtype);
        if (f_gu != null and f_dn != null) {
            exec_gu = moe_cpu_executor.Executor.initFull(gpa, .{
                .fmt = f_gu.?,
                .k_dim = spec.gate.in_dim,
                .out_dim = spec.gate.out_dim / 2, // banco fusionado gate|up por mitad
                .n_experts = spec.gate.n_expert,
            }) catch null;
            exec_dn = moe_cpu_executor.Executor.initFull(gpa, .{
                .fmt = f_dn.?,
                .k_dim = spec.down.in_dim,
                .out_dim = spec.down.out_dim,
                .n_experts = spec.down.n_expert,
            }) catch null;
            if (exec_gu != null and exec_dn != null) {
                _ = exec_gu.?.attachStream(@intFromPtr(stream));
                _ = exec_dn.?.attachStream(@intFromPtr(stream));
                for (layers.items) |*l| l.attachExecutors(exec_gu.?, exec_dn.?);
                std.debug.print("[moe_bench][lane-f] HÍBRIDO activo — executors attached ({d} capas)\n", .{layers.items.len});
            } else std.debug.print("[moe_bench][lane-f] fallo init executor → offload puro\n", .{});
        } else std.debug.print("[moe_bench][lane-f] dtype de bancos no soportado por GEMV CPU → offload puro\n", .{});
    }

    const n_embd: u32 = @intCast(spec.router.n_embd);
    var rng = std.Random.Xoshiro256.init(0xBEEF);
    const skew_mode = blk: {
        const v = std.c.getenv("MOE_SKEW") orelse break :blk false;
        break :blk std.mem.eql(u8, std.mem.span(v), "zipf");
    };
    var router_host: ?[]u8 = null;
    defer if (router_host) |r| gpa.free(r);
    if (skew_mode) {
        router_host = try gpa.alloc(u8, spec.router.bytes.len);
        @memcpy(router_host.?, spec.router.bytes);
    }
    // x persistente: en modo skew se regenera por paso (Zipf sobre expertos).
    const x_host = try gpa.alloc(f32, n_embd);
    defer gpa.free(x_host);
    for (x_host) |*v| v.* = rng.random().float(f32) - 0.5;
    const x_dev = try cudaz.cuMemAlloc(n_embd * @sizeOf(f32));
    defer cudaz.cuMemFree(x_dev);
    try moe_cuda.htod(f32, x_dev, x_host);
    const out_dev = try cudaz.cuMemAlloc(n_embd * @sizeOf(f32));
    defer cudaz.cuMemFree(out_dev);

    // Warm-up (llenar slots) + timing.
    for (layers.items, 0..) |*l, li_warm| {
        l.forwardGPU(x_dev, out_dev) catch |err| {
            std.debug.print("[lane-f] DIAG warmup capa {d}: {s}\n", .{ li_warm, @errorName(err) });
            return err;
        };
    }
    try cudaz.cuStreamSynchronize(stream);

    const steps: u32 = blk: {
        const arg = std.c.getenv("MOE_STEPS");
        if (arg) |a| {
            if (std.fmt.parseInt(u32, std.mem.span(a), 10)) |n| break :blk n else |_| {}
        }
        break :blk 100;
    };

    const ev0 = try cudaz.cuEventCreate(0);
    defer cudaz.cuEventDestroy(ev0);
    const ev1 = try cudaz.cuEventCreate(0);
    defer cudaz.cuEventDestroy(ev1);

    try cudaz.cuEventRecord(ev0, stream);
    for (0..steps) |step_i| {
        if (skew_mode) {
            const rf: []const f32 = @ptrCast(@alignCast(router_host.?));
            makeBiasedX(&rng, rf, n_embd, zipfTarget(rng.random(), info.n_expert), x_host);
            try moe_cuda.htod(f32, x_dev, x_host);
        }
        _ = step_i;
        for (layers.items) |*l| try l.forwardGPU(x_dev, out_dev);
    }
    try cudaz.cuEventRecord(ev1, stream);
    try cudaz.cuStreamSynchronize(stream);
    var ms: f32 = 0;
    try cudaz.cuEventElapsedTime(&ms, ev0, ev1);

    const n_calls: u64 = @as(u64, steps) * layers.items.len;
    const per_call = ms / @as(f32, @floatFromInt(n_calls));
    std.debug.print("[moe_bench] ff MoE bs=1 GATHER fused: {d:.3} ms/llamada-capa ({d} pasos × {d} capas, total {d:.1} ms)\n", .{ per_call, steps, layers.items.len, ms });

    // ── A3: misma carga con staging clásico ⇒ delta del gather fusionado ──
    if (!moe_cuda.noGatherEnabled()) {
        moe_cuda.g_force_classic = true;
        defer moe_cuda.g_force_classic = false;
        var s0 = try moe_cuda.readStats(gpa, &cache_gpu);
        defer s0.deinit(gpa);
        try cudaz.cuEventRecord(ev0, stream);
        for (0..steps) |_| {
            for (layers.items) |*l| try l.forwardGPU(x_dev, out_dev);
        }
        try cudaz.cuEventRecord(ev1, stream);
        try cudaz.cuStreamSynchronize(stream);
        var ms2: f32 = 0;
        try cudaz.cuEventElapsedTime(&ms2, ev0, ev1);
        const per_call2 = ms2 / @as(f32, @floatFromInt(n_calls));
        const speedup = if (ms2 > 0) ms2 / @max(ms, 0.001) else 0;
        std.debug.print("[moe_bench] ff MoE bs=1 STAGING clásico: {d:.3} ms/llamada-capa ⇒ gather es {d:.2}× más rápido\n", .{ per_call2, speedup });
    }

    // ── E6: stats DEVICE (contadores acumulados en kernel, lectura única) ──
    var gs = try moe_cuda.readStats(gpa, &cache_gpu);
    defer gs.deinit(gpa);
    var tot_active: i64 = 0;
    var tot_missing: i64 = 0;
    var tot_fetched: i64 = 0;
    for (gs.active_layer, 0..) |a_, li| {
        const m_ = gs.missing_layer[li];
        const f_ = gs.fetched_layer[li];
        const s_ = gs.steps_layer[li];
        if (s_ == 0) continue;
        tot_active += a_;
        tot_missing += m_;
        tot_fetched += f_;
        const mr = @as(f64, @floatFromInt(m_)) / @max(1, @as(f64, @floatFromInt(a_)));
        const fr = if (m_ == 0) 0 else @as(f64, @floatFromInt(f_)) / @as(f64, @floatFromInt(m_));
        std.debug.print("[moe_bench] capa {d}: pasos={d} activos={d} misses={d} fetch={d} (miss={d:.1}% fetch={d:.1}%)\n", .{ li, s_, a_, m_, f_, mr * 100, fr * 100 });
    }
    const dev_miss = if (tot_active == 0) 0 else @as(f64, @floatFromInt(tot_missing)) / @as(f64, @floatFromInt(tot_active));
    const dev_fetch = if (tot_missing == 0) 0 else @as(f64, @floatFromInt(tot_fetched)) / @as(f64, @floatFromInt(tot_missing));
    const dev_hit = 1.0 - dev_miss;

    // Oracle: cota de hit-rate con este cache_size dado el histograma real.
    const freq = try moe_cuda.readDecodeFreq(gpa, &cache_gpu);
    defer gpa.free(freq);
    const oracle = moe_cuda.oracleHitAtSlots(freq, cache_gpu.cfg.cache_size);
    const ratio = if (oracle > 0) dev_hit / oracle else 0;
    std.debug.print("[moe_bench] DEVICE: hit_rate={d:.3} miss_rate={d:.3} fetch_rate={d:.3} | oracle_hit_at_{d}={d:.3} | ratio={d:.2} (criterio ≥0.70)\n", .{ dev_hit, dev_miss, dev_fetch, cache_gpu.cfg.cache_size, oracle, ratio });
}
