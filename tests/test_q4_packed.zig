//! Test 1.3 (lane-f): paridad bit-exacta del GEMV q4_0 packed (dual-view
//! payload|d) vs el dp4a canónico (§5.8), ambas rutas leyendo el peso desde
//! DEVICE (cache q4Weight/q4PackedWeight — comparación de kernel, no de PCIe).
//!
//! Resultado de la investigación 1.3: el kernel packed NO acelera (1.00-1.05x,
//! BW idéntico ~164 GB/s en N=6144) porque el cuello es el total de sectores
//! HBM por warp, que es invariante al layout (576B/32SB = 18 sectores either
//! way). Este test protege la CORRECTITUD del camino experimental Q4PACK=1
//! (por si un formato futuro sí cambia el conteo de sectores) y la del
//! pre-warm que evita alloc/H2D dentro de CUDA-graph capture.
//! Requiere CUDA; sin GPU sale limpio.
const std = @import("std");
const matmul = @import("matmul");
const layer_kernels = @import("layer_kernels");
const cublas = @import("cublas");
const cudaz = @import("cudaz");
const T = @import("time");
const now = T.Timer.now;

test "q4-packed: M1 dual-view paridad bit-exacta vs dp4a canónico" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const allocator = std.heap.page_allocator;
    var lk = try layer_kernels.LayerKernels.init(@ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw));
    defer lk.deinit();

    const K: usize = 1024;
    var rng = std.Random.DefaultPrng.init(1234);
    const random = rng.random();

    var a_host: [1024]f32 = undefined;
    for (&a_host) |*v| v.* = random.floatNorm(f32) * 1.2;

    const kb = K / 32;
    // Geometrías reales del Qwen3.5-0.8B (z/qkv/ffn_gate del ΔNet layer).
    inline for (.{ 2048, 3584, 6144 }) |N| {
        const wbytes_len = N * kb * 18;
        const w_bytes = try allocator.alloc(u8, wbytes_len);
        defer allocator.free(w_bytes);
        for (w_bytes) |*b| b.* = random.int(u8);
        // Escalas d f16 válidas (half 0x2C00 = 0.25): el payload aleatorio
        // produce quanta arbitrarias, la escala sólo evita NaN/Inf.
        for (0..N) |r| {
            for (0..kb) |kbi| {
                const off = (r * kb + kbi) * 18;
                w_bytes[off] = 0x00;
                w_bytes[off + 1] = 0x2C;
            }
        }

        const g_a = try cublas.GpuTensor(f32).alloc(K);
        defer g_a.deinit();
        const g_c1 = try cublas.GpuTensor(f32).alloc(N);
        defer g_c1.deinit();
        const g_c2 = try cublas.GpuTensor(f32).alloc(N);
        defer g_c2.deinit();
        try g_a.buf.upload(&a_host);

        // Ambos caminos leen el peso desde DEVICE: canónico vía q4Weight,
        // packed vía su cache propio — así la paridad mide kernel, no PCIe.
        const dev_w = try layer_kernels.q4Weight(allocator, @intFromPtr(w_bytes.ptr), w_bytes);
        try lk.q4gemmM1Dp4a(g_a.ptr(), dev_w, g_c1.ptr(), K, N);
        try lk.q4gemmM1Dp4aPacked(allocator, g_a.ptr(), w_bytes, g_c2.ptr(), K, N);
        try cudaz.cuStreamSynchronize(lk.stream);

        const h1 = try allocator.alloc(f32, N);
        defer allocator.free(h1);
        const h2 = try allocator.alloc(f32, N);
        defer allocator.free(h2);
        try g_c1.buf.download(h1);
        try g_c2.buf.download(h2);
        var max_abs: f32 = 0;
        for (h1, h2) |x, y| {
            const d = @abs(x - y);
            if (d > max_abs) max_abs = d;
        }
        // Mismo contenido, distinta dirección: la matemática es idéntica y
        // el orden de reducción también → bit-exacto por construcción.
        try std.testing.expectEqual(@as(f32, 0), max_abs);
    }
}

test "q4-packed: repack dual-view roundtrip (host)" {
    // El repack separa payload (16B/SB contiguos) y escalas d (f16 array).
    // Roundtrip inverso → bytes canónicos exactos.
    const allocator = std.heap.page_allocator;
    var rng = std.Random.DefaultPrng.init(99);
    const random = rng.random();
    const K: usize = 1024;
    const N: usize = 64;
    const kb = K / 32;
    const src = try allocator.alloc(u8, N * kb * 18);
    defer allocator.free(src);
    for (src) |*b| b.* = random.int(u8);

    const rep = try layer_kernels.repackQ4PayloadD(src, N, K, allocator);
    defer allocator.free(rep.payload);
    defer allocator.free(rep.d);

    // Inverso: re-ensamblar SB canónicos [d f16][16B quanta].
    const back = try allocator.alloc(u8, N * kb * 18);
    defer allocator.free(back);
    for (0..N) |r| {
        for (0..kb) |kbi| {
            const i = (r * kb + kbi) * 18;
            const d_u16: u16 = @bitCast(rep.d[r * kb + kbi]);
            back[i] = @truncate(d_u16);
            back[i + 1] = @truncate(d_u16 >> 8);
            @memcpy(back[i + 2 ..][0..16], rep.payload[(r * kb + kbi) * 16 ..][0..16]);
        }
    }
    try std.testing.expectEqualSlices(u8, src, back);
}
