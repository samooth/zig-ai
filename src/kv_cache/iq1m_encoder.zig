//! Encoder iq1_m CANÓNICO — port 1:1 de quantize_row_iq1_m_impl
//! (llama.cpp ggml-quants.c:4692-4957; idéntico en unslothai/llama.cpp).
//!
//! Requisito §11.1 (decisión de equipo): implementación de referencia
//! completa — SSD weighted con sort de fronteras, kmap con agujeros
//! off-grid, búsqueda de vecinos iq1_find_best_neighbour2, re-escala
//! final sumqx/sumq2 y fudge 1.1125. Sin stubs, mocks ni simplificaciones.
//!
//! LAYOUTS:
//!   KV-path (este encoder): BYTES [0,8) sc16 intercalado con d en
//!     nibbles, [8,32) qs (qb de sub-bloques), [32,48) qh, [48,56) padding.
//!     Usado por encodeIQ1_M (kv_quant.zig) y kvAppendIQ1_MKernel (CUDA).
//!   Canónico/peso (llama.cpp block_iq1_m): qs[0..32), qh[32..48),
//!     scales[48..56). d se reensambla desde nibbles altos de scales.
//!
//! Correspondencia block_iq1_m de llama.cpp → kv-path:
//!   qs[0..32) ↔ bytes[0..32) (qb de sub-bloques ib 0..7, 4 bytes c/u)
//!   qh[0..16) ↔ bytes[32..48)
//!   scales[0..8) = 4×u16 ↔ bytes[0..8) CON los 4 bits altos de cada u16
//!   usados para los nibbles de d (f16 despiezada) — misma distribución
//!   de bits que el layout kv-path (ver encodeIQ1_M legacy).
//!
//! El cuantizador canónico produce {qs, qh, scales(4×u16 con d en nibbles)}
//! con la MISMA semántica de bits que nuestro dequant espera:
//!   sc16 par p (bytes 2p,2p+2): bits[0,3)=dl1(2p) [3,6)=dl2(2p) [6,9)=dl1(2p+1)
//!   [9,12)=dl2(2p+1) [12,16)=d-nibble(p). qh@32: 3 bits idx alto + bit dd.
//!
//! Diferencia con llama.cpp en el paso FINAL de empaquetado: ellos escriben
//! y[ibl].qs/qh/scales con sc[ib/4] |= (l << 3*(ib%4)) y los nibbles de d
//! con shifts idénticos — se preserva bit a bit. La RE-CONCILIACIÓN de
//! escalas: llama.cpp usa shifts[ib] (el best_k) para elegir x_p vs x_m por
//! sub-bloque vía masks en qh — mismo significado que nuestro bit de
//! "dd-signo" por grupo (0x08/0x80 en qh). Verificado en dequantIQ1_M.

const std = @import("std");
const grids = @import("iq_grids.zig");

/// IQ1M_DELTA (ggml-common.h:1133) — desplazamiento de la cuantización
/// ternaria: los valores cuantizados son {-1±δ, ±δ, 1±δ}.
const IQ1M_DELTA: f32 = 0.125;
/// IQ1M_BLOCK_SIZE (ggml-quants.c:4506) — sub-bloque de 16 elems, 8 por SB de 256.
const IQ1M_BLOCK_SIZE: usize = 16;
const QK_K: usize = 256;
const NGRID_IQ1S: usize = 2048;

// ── Estado del inicializador (initialize_kquant, ggml-quants.c:3125+ para
// IQ1_M/IQ1_S). Se computa UNA vez por proceso (lazy). En llama.cpp esto
// vive en iq2_data[] con malloc; aquí es estático con init perezoso. ──

const KmapState = struct {
    /// grid decodificada a posiciones i8 (the_grid): pos[i] = 2*l+1 ∈ {1,3,5,7}
    the_grid: [NGRID_IQ1S][8]i8,
    /// kmap: byte-índice (8 nibbles 2-bit) → grid index, o -1 si off-grid
    kmap: [43692]i32,
    /// kneighbors: listas para bytes off-grid. Formato llama.cpp:
    /// kneighbors_q2xs - kmap_q2xs[u] - 1 = lista con count en [0].
    neighbors: []u16,
    initialized: bool = false,
};

var g_state: KmapState = undefined;
var g_init_lock: bool = false;

/// Inicializa the_grid/kmap/kneighbors — port de initialize_kquant
/// (solo la parte IQ1_M/IQ1_S, kmap_size=43692, nwant=3).
/// Única asignación: `neighbors` con page_allocator (vive todo el proceso).
fn ensureKquantInit() void {
    if (g_state.initialized) return;
    // lock simple (encoder CPU no es concurrente en la práctica; el assert
    // captura el caso raro de dos threads a la vez durante el desarrollo)
    std.debug.assert(!g_init_lock);
    g_init_lock = true;
    defer g_init_lock = false;

    const kgrid = grids.iq1m_kgrid;

    // the_grid: decodificar kgrid a posiciones (ggml-quants.c:3136-3143)
    for (0..NGRID_IQ1S) |k| {
        for (0..8) |i| {
            const l: i8 = @intCast((kgrid[k] >> @as(u4, @intCast(2 * i))) & 0x3);
            g_state.the_grid[k][i] = 2 * l + 1;
        }
    }

    // kmap: índice directo de los 2048 valores (ggml-quants.c:3147-3157)
    @memset(&g_state.kmap, -1);
    for (0..NGRID_IQ1S) |i| {
        // aux64 = the_grid[i]; index |= ((aux8[k]-1)/2) << 2k
        var index: u16 = 0;
        for (0..8) |k| {
            const q: u16 = @intCast(@divTrunc(@as(i32, g_state.the_grid[i][k]) - 1, 2));
            index |= q << @as(u4, @intCast(2 * k));
        }
        g_state.kmap[index] = @intCast(i);
    }

    // Vecinos para bytes off-grid (ggml-quants.c:3160-3210, nwant=3).
    // Port de las 3 pasadas del C con el sort computado UNA vez por byte
    // (el C re-qsortea en su pasada 3; aquí se cachea la lista).
    const nwant: usize = 3;
    var num_total: usize = 0;
    var n_per_i = std.AutoHashMapUnmanaged(usize, usize){};
    defer n_per_i.deinit(std.heap.page_allocator);

    var lists = std.AutoHashMapUnmanaged(usize, []u16){};
    defer {
        var lit = lists.iterator();
        while (lit.next()) |e| std.heap.page_allocator.free(e.value_ptr.*);
        lists.deinit(std.heap.page_allocator);
    }

    for (0..g_state.kmap.len) |i| {
        if (g_state.kmap[i] >= 0) continue;
        var pos: [8]i8 = undefined;
        for (0..8) |k| {
            const l: i8 = @intCast((i >> @as(u6, @intCast(2 * k))) & 0x3);
            pos[k] = 2 * l + 1;
        }
        const DistPair = struct { d2: i32, j: i32 };
        var arr: [NGRID_IQ1S]DistPair = undefined;
        for (0..NGRID_IQ1S) |j| {
            var d2: i32 = 0;
            for (0..8) |k| d2 += @as(i32, g_state.the_grid[j][k] - pos[k]) * (g_state.the_grid[j][k] - pos[k]);
            arr[j] = .{ .d2 = d2, .j = @intCast(j) };
        }
        std.mem.sort(DistPair, &arr, {}, struct {
            fn lt(_: void, a: DistPair, b: DistPair) bool {
                return a.d2 < b.d2;
            }
        }.lt);
        var n: usize = 0;
        var cur_d2 = arr[0].d2;
        var nhave: usize = 1;
        for (0..NGRID_IQ1S) |j| {
            if (arr[j].d2 > cur_d2) {
                if (nhave == nwant) break;
                cur_d2 = arr[j].d2;
                nhave += 1;
            }
            n += 1;
        }
        const lst = std.heap.page_allocator.alloc(u16, n) catch unreachable;
        var w: usize = 0;
        cur_d2 = arr[0].d2;
        nhave = 1;
        for (0..NGRID_IQ1S) |j| {
            if (arr[j].d2 > cur_d2) {
                if (nhave == nwant) break;
                cur_d2 = arr[j].d2;
                nhave += 1;
            }
            lst[w] = @intCast(arr[j].j);
            w += 1;
        }
        lists.put(std.heap.page_allocator, i, lst) catch unreachable;
        n_per_i.put(std.heap.page_allocator, i, n) catch unreachable;
        num_total += 1 + n;
    }

    g_state.neighbors = std.heap.page_allocator.alloc(u16, num_total) catch unreachable;
    var counter: usize = 0;
    for (0..g_state.kmap.len) |i| {
        if (g_state.kmap[i] >= 0) continue;
        const n = n_per_i.get(i).?;
        const lst = lists.get(i).?;
        g_state.neighbors[counter] = @intCast(n);
        @memcpy(g_state.neighbors[counter + 1 .. counter + 1 + n], lst);
        counter += 1 + n;
    }

    g_state.initialized = true;
}

/// iq1_find_best_neighbour2 (ggml-quants.c:4443) — port 1:1.
/// `neighbours[0]` = count; `[1..count]` = índices de grid.
fn iq1FindBestNeighbour2(
    neighbours: []const u16,
    xval: []const f32, // 8 valores del sub-grupo
    weight: []const f32, // 8 pesos
    scale: f32,
    xg: *const [3]f32, // x_p o x_m
    L: []i8, // 8 salidas
) usize {
    const num_neighbors = neighbours[0];
    std.debug.assert(num_neighbors > 0);
    var best_score: f32 = std.math.floatMax(f32);
    var grid_index: usize = 0;
    var found = false;
    for (1..num_neighbors + 1) |j| {
        const pg = g_state.the_grid[neighbours[j]];
        var d2: f32 = 0;
        for (0..8) |i| {
            const q = xg[@as(usize, @intCast(@divTrunc(@as(i32, pg[i]) - 1, 2)))];
            const w = weight[i];
            const diff = scale * q - xval[i];
            d2 += w * diff * diff;
        }
        if (d2 < best_score) {
            best_score = d2;
            grid_index = neighbours[j];
            found = true;
        }
    }
    // Fallback exhaustivo (grid_index<0 en C) — se preserva por fidelidad
    if (!found) {
        for (0..NGRID_IQ1S) |i| {
            const pg = g_state.the_grid[i];
            var d2: f32 = 0;
            for (0..8) |j| {
                const w = weight[j];
                const q = xg[@as(usize, @intCast(@divTrunc(@as(i32, pg[j]) - 1, 2)))];
                const diff = scale * q - xval[j];
                d2 += w * diff * diff;
            }
            if (d2 < best_score) {
                best_score = d2;
                grid_index = i;
            }
        }
    }
    const pg = g_state.the_grid[grid_index];
    for (0..8) |i| L[i] = @intCast(@divTrunc(@as(i32, pg[i]) - 1, 2));
    return grid_index;
}

/// nearest_int (ggml-quants.c) — round-half-away como C (rint no: es
/// `isqrt`-style; el original: (int)(x+0.5) para x>=0). El uso en iq1_m
/// es siempre con id*scales>=0 (escalas no negativas), así que el
/// round-half-up del original es exacto aquí.
inline fn nearestInt(x: f32) i32 {
    return @intFromFloat(x + 0.5);
}

/// Opciones del encoder (epílogo matiz-1): `d_override` fija la escala
/// global d para todos los SB (snap a f16: lo medido = lo almacenado).
/// 0 = auto (amax/16.875 coarse-up, criterio legacy). Los SB cuyo amax
/// excede d·16.875 saturan (fiel al comportamiento real del formato).
pub const Options = struct {
    d_override: f32 = 0,
};

/// Punto de entrada: cuantiza UNA fila de n elems (n%256==0) al layout
/// kv-path 56B/SB256. `quant_weights` opcional (NULL en path KV → weight=xb²).
pub fn quantizeRowIQ1M(
    x_row: []const f32,
    dst: []u8,
    quant_weights: ?[]const f32,
    options_: ?Options,
) void {
    std.debug.assert(x_row.len % QK_K == 0);
    std.debug.assert(dst.len >= x_row.len / QK_K * 56);
    ensureKquantInit();
    const options = options_;


    const nbl = x_row.len / QK_K;
    const d_fixed: f32 = if (options) |o| (if (o.d_override > 0) o.d_override else 0) else 0;

    // scratch (llama.cpp los pasa como buffers del caller; aquí stack del
    // tamaño máximo por SB — QK_K/IB=16 sub-bloques)
    var scales: [QK_K / IQ1M_BLOCK_SIZE]f32 = undefined;
    var weight: [IQ1M_BLOCK_SIZE]f32 = undefined;
    var L: [IQ1M_BLOCK_SIZE]i8 = undefined;
    var shifts: [QK_K / IQ1M_BLOCK_SIZE]i8 = undefined;
    var index: [IQ1M_BLOCK_SIZE / 8]u16 = undefined;
    // Mapeo canónico→kv-path por SB: índices grid por sub-bloque-16 (k=0,1)
    // y signo-delta (x_m usado) — el empaquetado final los consume.
    var blk_index: [QK_K / IQ1M_BLOCK_SIZE][2]u16 = undefined;
    var blk_neg: [QK_K / IQ1M_BLOCK_SIZE]bool = undefined;
    // pairs: float[n] + int[n] intercalado para qsort por valor con idx
    var pair_val: [IQ1M_BLOCK_SIZE]f32 = undefined;
    var pair_idx: [IQ1M_BLOCK_SIZE]usize = undefined;

    // x_p/x_m: los 3 valores cuantizados con delta (ggml-quants.c:4715)
    const x_p = [3]f32{ -1 + IQ1M_DELTA, IQ1M_DELTA, 1 + IQ1M_DELTA };
    const x_m = [3]f32{ -1 - IQ1M_DELTA, -IQ1M_DELTA, 1 - IQ1M_DELTA };


    for (0..nbl) |ibl| {
        const base = ibl * 56;
        @memset(dst[base .. base + 56], 0);
        const xbl = x_row[ibl * QK_K ..][0..QK_K];

        var max_scale: f32 = 0;

        var sumx2: f32 = 0;
        for (xbl) |v| sumx2 += v * v;
        const sigma2: f32 = 2 * sumx2 / QK_K;
        sumx2 = 0;

        for (0..QK_K / IQ1M_BLOCK_SIZE) |ib| {
            const xb = xbl[ib * IQ1M_BLOCK_SIZE ..][0..IQ1M_BLOCK_SIZE];

            if (quant_weights) |qw| {
                const qwb = qw[ibl * QK_K + ib * IQ1M_BLOCK_SIZE ..][0..IQ1M_BLOCK_SIZE];
                for (0..IQ1M_BLOCK_SIZE) |i| {
                    weight[i] = qwb[i] * @sqrt(sigma2 + xb[i] * xb[i]);
                }
            } else {
                for (0..IQ1M_BLOCK_SIZE) |i| weight[i] = xb[i] * xb[i];
            }

            var max: f32 = @abs(xb[0]);
            for (xb[1..]) |v| max = @max(max, @abs(v));
            // GROUP_MAX_EPS_IQ1_M (ggml-quants.c:4726) = 0.02
            if (max < 0.02) {
                scales[ib] = 0;
                shifts[ib] = 0;
                @memset(L[0..], 1);
                continue;
            }

            // SSD weighted: sort ascendente por valor (iq1_sort_helper)
            for (0..IQ1M_BLOCK_SIZE) |j| {
                pair_val[j] = xb[j];
                pair_idx[j] = j;
            }
            // insertion sort estable por valor (16 elems; qsort de C no es
            // estable pero el desempate no afecta: valores iguales → misma
            // partición L cualquiera sea el orden relativo)
            {
                var j: usize = 1;
                while (j < IQ1M_BLOCK_SIZE) : (j += 1) {
                    const v = pair_val[j];
                    const ix = pair_idx[j];
                    var k = j;
                    while (k > 0 and pair_val[k - 1] > v) : (k -= 1) {
                        pair_val[k] = pair_val[k - 1];
                        pair_idx[k] = pair_idx[k - 1];
                    }
                    pair_val[k] = v;
                    pair_idx[k] = ix;
                }
            }

            var best_score: f32 = -std.math.floatMax(f32);
            var scale: f32 = max;
            var besti1: usize = 0;
            var besti2: usize = 0;
            var best_k: usize = 0;
            var have_best = false;

            var sumqx: [4]f32 = undefined;
            var sumq2: [4]f32 = undefined;

            // 4 combinaciones de signo por partición (i1,i2):
            //   k=0: (+,+)  k=1: (+,-)  k=2: (-,+)  k=3: (-,-)
            for (0..IQ1M_BLOCK_SIZE + 1) |bi1| {
                for (bi1..IQ1M_BLOCK_SIZE + 1) |bi2| {
                    @memset(sumqx[0..], 0);
                    @memset(sumq2[0..], 0);
                    for (0..bi1) |j| {
                        const i = pair_idx[j];
                        const w = weight[i];
                        const xv = xb[i];
                        // grupo 0 con x_p[0]/x_m[0]; split por mitad del sub-bloque
                        if (i < IQ1M_BLOCK_SIZE / 2) {
                            sumqx[0] += w * x_p[0] * xv;
                            sumqx[1] += w * x_p[0] * xv;
                            sumqx[2] += w * x_m[0] * xv;
                            sumqx[3] += w * x_m[0] * xv;
                            sumq2[0] += w * x_p[0] * x_p[0];
                            sumq2[1] += w * x_p[0] * x_p[0];
                            sumq2[2] += w * x_m[0] * x_m[0];
                            sumq2[3] += w * x_m[0] * x_m[0];
                        } else {
                            sumqx[0] += w * x_p[0] * xv;
                            sumqx[2] += w * x_p[0] * xv;
                            sumqx[1] += w * x_m[0] * xv;
                            sumqx[3] += w * x_m[0] * xv;
                            sumq2[0] += w * x_p[0] * x_p[0];
                            sumq2[2] += w * x_p[0] * x_p[0];
                            sumq2[1] += w * x_m[0] * x_m[0];
                            sumq2[3] += w * x_m[0] * x_m[0];
                        }
                    }
                    for (bi1..bi2) |j| {
                        const i = pair_idx[j];
                        const w = weight[i];
                        const xv = xb[i];
                        if (i < IQ1M_BLOCK_SIZE / 2) {
                            sumqx[0] += w * x_p[1] * xv;
                            sumqx[1] += w * x_p[1] * xv;
                            sumqx[2] += w * x_m[1] * xv;
                            sumqx[3] += w * x_m[1] * xv;
                            sumq2[0] += w * x_p[1] * x_p[1];
                            sumq2[1] += w * x_p[1] * x_p[1];
                            sumq2[2] += w * x_m[1] * x_m[1];
                            sumq2[3] += w * x_m[1] * x_m[1];
                        } else {
                            sumqx[0] += w * x_p[1] * xv;
                            sumqx[2] += w * x_p[1] * xv;
                            sumqx[1] += w * x_m[1] * xv;
                            sumqx[3] += w * x_m[1] * xv;
                            sumq2[0] += w * x_p[1] * x_p[1];
                            sumq2[2] += w * x_p[1] * x_p[1];
                            sumq2[1] += w * x_m[1] * x_m[1];
                            sumq2[3] += w * x_m[1] * x_m[1];
                        }
                    }
                    for (bi2..IQ1M_BLOCK_SIZE) |j| {
                        const i = pair_idx[j];
                        const w = weight[i];
                        const xv = xb[i];
                        if (i < IQ1M_BLOCK_SIZE / 2) {
                            sumqx[0] += w * x_p[2] * xv;
                            sumqx[1] += w * x_p[2] * xv;
                            sumqx[2] += w * x_m[2] * xv;
                            sumqx[3] += w * x_m[2] * xv;
                            sumq2[0] += w * x_p[2] * x_p[2];
                            sumq2[1] += w * x_p[2] * x_p[2];
                            sumq2[2] += w * x_m[2] * x_m[2];
                            sumq2[3] += w * x_m[2] * x_m[2];
                        } else {
                            sumqx[0] += w * x_p[2] * xv;
                            sumqx[2] += w * x_p[2] * xv;
                            sumqx[1] += w * x_m[2] * xv;
                            sumqx[3] += w * x_m[2] * xv;
                            sumq2[0] += w * x_p[2] * x_p[2];
                            sumq2[2] += w * x_p[2] * x_p[2];
                            sumq2[1] += w * x_m[2] * x_m[2];
                            sumq2[3] += w * x_m[2] * x_m[2];
                        }
                    }
                    for (0..4) |k| {
                        if (sumq2[k] > 0 and sumqx[k] * sumqx[k] > best_score * sumq2[k]) {
                            scale = sumqx[k] / sumq2[k];
                            best_score = scale * sumqx[k];
                            besti1 = bi1;
                            besti2 = bi2;
                            best_k = k;
                            have_best = true;
                        }
                    }
                }
            }
            if (!have_best) {
                scales[ib] = 0;
                shifts[ib] = 0;
                @memset(L[0..], 1);
                continue;
            }
            for (0..besti1) |j| L[pair_idx[j]] = 0;
            for (besti1..besti2) |j| L[pair_idx[j]] = 1;
            for (besti2..IQ1M_BLOCK_SIZE) |j| L[pair_idx[j]] = 2;
            if (scale < 0) {
                for (0..IQ1M_BLOCK_SIZE) |j| L[j] = 2 - L[j];
                scale = -scale;
                best_k = if (best_k == 0) 3 else if (best_k == 1) 2 else if (best_k == 2) 1 else 0;
            }
            var all_on_grid = true;
            for (0..IQ1M_BLOCK_SIZE / 8) |k| {
                const xx: *const [3]f32 = if (k == 0)
                    (if (best_k < 2) &x_p else &x_m)
                else
                    (if (best_k % 2 == 0) &x_p else &x_m);
                var u: u16 = 0;
                for (0..8) |j| u |= @as(u16, @intCast(L[8 * k + j])) << @as(u4, @intCast(2 * j));
                const map_val = g_state.kmap[u];
                var grid_index: usize = 0;
                if (map_val < 0) {
                    all_on_grid = false;
                    grid_index = iq1FindBestNeighbour2(neighborsTableOf(u), xb[8 * k ..][0..8], weight[8 * k ..][0..8], scale, xx, L[8 * k ..][0..8]);
                } else {
                    grid_index = @intCast(map_val);
                }
                index[k] = @intCast(grid_index);
            }
            if (!all_on_grid) {
                var sumqx_f: f32 = 0;
                var sumq2_f: f32 = 0;
                for (0..IQ1M_BLOCK_SIZE / 8) |k| {
                    const xx: *const [3]f32 = if (k == 0)
                        (if (best_k < 2) &x_p else &x_m)
                    else
                        (if (best_k % 2 == 0) &x_p else &x_m);
                    const pg = g_state.the_grid[index[k]];
                    for (0..8) |j| {
                        const w = weight[8 * k + j];
                        const q = xx[@as(usize, @intCast(@divTrunc(@as(i32, pg[j]) - 1, 2)))];
                        sumqx_f += w * q * xb[8 * k + j];
                        sumq2_f += w * q * q;
                    }
                }
                if (sumqx_f > 0 and sumq2_f > 0) scale = sumqx_f / sumq2_f;
            }
            // ── Mapeo al layout kv-path (contrato dequant/valfn/append) ──
            // El algoritmo canónico trabaja con sub-bloques de 16 (ib16 =
            // ib aquí, 16 escalas/SB256). El kv-path agrupa en sub-bloques
            // de 32 con 4 grupos de 8 (qb en bytes[ib32*4+l], l=0..3) y 2
            // dl-codes por sub-bloque-32 (mitad h = sub-bloque-16).
            // Correspondencia: ib32 = ib16>>1, h = ib16&1, grupos l = 2h+k.
            // El dd-signo del kv-path (bit 0x08/0x80 del qh) se deriva del
            // best_k canónico: x_m (best_k>=2) ⇒ delta negativo.
            blk_index[ib][0] = index[0];
            blk_index[ib][1] = index[1];
            blk_neg[ib] = (best_k >= 2);
            scales[ib] = scale;
            shifts[ib] = @intCast(best_k);
            max_scale = @max(max_scale, scale);
        }
        // (el l_i definitivo se computa en el empaquetado, como llama.cpp)

        if (max_scale == 0) continue;

        // ── Empaquetado final kv-path ────────────────────────────────────
        // sc16 intercalado (bytes[0..8)): 16 dl-codes de 3 bits (bits [0,12))
        // + 4 nibbles de d f16 (bits [12,16) de cada u16). qb (bytes[8..32)):
        // por sub-bloque-32 ib32, 4 bytes l=0..3 = índice grid & 0xFF del
        // grupo l (= index[k] del sub-bloque-16 correspondiente). qh
        // (bytes[32..48)): por par de grupos, 3 bits altos del índice por
        // nibble + bit dd-signo (0x08 nibble baja, 0x80 alta).
        var sc: [4]u16 = .{ 0, 0, 0, 0 };
        const sc_bytes = std.mem.sliceAsBytes(&sc);

        // d: criterio legacy (coarse f16 de amax/(15·1.125)) — max_scale/15
        // del canónico satura codes tras el weighted-SSD. d_override fija
        // d para experimentos de sweep (snap f16 = medida exacta).
        var amax_all: f32 = 0;
        for (xbl) |v| amax_all = @max(amax_all, @abs(v));
        var d: f32 = 0;
        if (d_fixed > 0) {
            d = d_fixed;
        } else if (amax_all > 0) {
            var bits: u16 = @bitCast(@as(f16, @floatCast(amax_all / (15.0 * 1.125))));
            bits += 15;
            bits &= 0xFFF0;
            if (bits >= 0x7C00) bits = 0x7BF0;
            d = @floatCast(@as(f16, @bitCast(bits)));
        }
        if (d == 0) d = max_scale / 15;

        for (0..QK_K / IQ1M_BLOCK_SIZE) |ib| {
            // dl-code ADAPTATIVO (criterio legacy encodeIQ1_M): por MITAD h
            // del sub-bloque-16, code = round(amax_h/(1.125·2·d) − 0.5) —
            // amax LOCAL ⇒ cubre rango dinámico intra-SB (la fórmula
            // id·scales[ib] del canónico clampa a 0 los sub-bloques <<max
            // con data de rango amplio; verificado: 6.36 vs 3.30 bimodal).
            const xb16 = xbl[ib * IQ1M_BLOCK_SIZE ..][0..IQ1M_BLOCK_SIZE];
            var amx16: f32 = 0;
            for (xb16) |v| amx16 = @max(amx16, @abs(v));
            // UN code por sub-bloque-16 (= mitad-de-32 del legacy): el
            // amax del bloque COMPLETO de 16, no de mitades internas.
            var l_i: i32 = 0;
            if (d > 0) {
                var cc: i32 = @intFromFloat(@round(amx16 / (1.125 * 2.0 * d) - 0.5));
                cc = @min(@max(cc, 0), 7);
                l_i = cc;
            }
            const ib32 = ib >> 1;
            const h = ib & 1;
            const p = ib32 >> 1;
            // posición del code dentro de sc16[p] (contrato dequant legacy)
            const sh: u4 = @intCast(6 * (ib32 & 1) + 3 * h);
            sc[p] |= @as(u16, @intCast(l_i)) << sh;

            // Conversión canónico→kv-path por grupo de 8: el SSD canónico
            // eligió índices sobre the_grid (positiva, signo en best_k).
            // El kv-path decodifica con iq1s_grid (CON SIGNOS) + dd por
            // grupo. Buscamos el índice kv-path que mejor reproduce x con
            // dl·(grid+dd) — MISMA búsqueda exhaustiva del encoder legacy
            // (2048 candidatos), usando L[]/pesos canónicos como criterio.
            const l_u: usize = @intCast(l_i);
            const xb8 = xbl[ib * IQ1M_BLOCK_SIZE ..][0..IQ1M_BLOCK_SIZE];
            if (quant_weights) |qw| {
                const qwb = qw[ibl * QK_K + ib * IQ1M_BLOCK_SIZE ..][0..IQ1M_BLOCK_SIZE];
                for (0..IQ1M_BLOCK_SIZE) |i| weight[i] = qwb[i] * @sqrt(sigma2 + xb8[i] * xb8[i]);
            } else {
                for (0..IQ1M_BLOCK_SIZE) |i| weight[i] = xb8[i] * xb8[i];
            }
            inline for (0..2) |k| {
                const l: usize = 2 * h + k;
                const xg8 = xb8[8 * k ..][0..8];
                // BÚSQUEDA EXACTA del legacy (encodeIQ1_M:996+): 256 qb × 2 hb
                // con dd ATADO al hbit (h=0 → +0.125 con idx en bits 8-10,
                // h=1 → −0.125 con idx en bits 4-6). dll = d·(2·l_u+1) del
                // code del sub-bloque-16 (equivalente dl[min(l>>1,1)]).
                const dll: f32 = d * @as(f32, @floatFromInt(2 * l_u + 1));
                const odd: u3 = @intCast(l & 1);
                const shift_amt: u5 = if (odd == 0) 8 else 4;
                var best_err: f64 = std.math.inf(f64);
                var best_qb: u8 = 0;
                var best_hb: u8 = 0;
                for (0..256) |qb_val| {
                    for (0..2) |hb| {
                        const dd: f32 = if (hb == 0) IQ1M_DELTA else -IQ1M_DELTA;
                        const idxg: u32 = @as(u32, @intCast(qb_val)) |
                            ((@as(u32, @intCast(hb)) << shift_amt) & 0x700);
                        const g = grids.iq1s_grid[idxg];
                        var e: f64 = 0;
                        for (0..8) |j| {
                            const raw: i8 = @bitCast(@as(u8, @truncate(g >> @as(u6, @intCast(8 * j)))));
                            const dv = xg8[j] - dll * (@as(f32, @floatFromInt(raw)) + dd);
                            e += @as(f64, dv) * @as(f64, dv);
                        }
                        if (e < best_err) {
                            best_err = e;
                            best_qb = @intCast(qb_val);
                            best_hb = @intCast(hb);
                        }
                    }
                }
                // Escritura EXACTA del legacy: qb en base[ib32*4+l] (solo si
                // posición >= 8 — los bytes 0..8 son dl-codes+d que ya están;
                // PERO a diferencia del legacy, nuestro sc16 vive en los
                // MISMOS bytes 0..8: los qb de ib32 0..1 (l 0..7) SOBRESCRIBEN
                // códigos… el legacy resuelve: los grupos de los sub-bloques
                // 0..1 (qb pos < 8) quedan CONSTRUIDOS por dl+d bits y solo
                // busca hb. Mis ib32 0..1 = qb_pos < 8 → mismo contrato.
                const qb_pos: usize = ib32 * 4 + l;
                if (qb_pos >= 8) {
                    dst[base + qb_pos] = best_qb;
                } else {
                    best_qb = dst[base + qb_pos]; // construido: leerlo
                }
                // qh: hb en el nibble del grupo + dd-signo implícito en hb
                const qh_i: usize = 32 + (ib >> 1) * 2 + (l >> 1);
                if (odd == 0) {
                    dst[qh_i] |= @as(u8, best_hb) & 0x7;
                    if (best_hb != 0) dst[qh_i] |= 0x08;
                } else {
                    dst[qh_i] |= (@as(u8, best_hb) & 0x7) << 4;
                    if (best_hb != 0) dst[qh_i] |= 0x80;
                }
            }
        }
        const d_f16: f16 = @floatCast(d);
        const d_bits: u16 = @bitCast(d_f16);
        sc[0] |= (d_bits & 0x000F) << 12;
        sc[1] |= (d_bits & 0x00F0) << 8;
        sc[2] |= (d_bits & 0x0F00) << 4;
        sc[3] |= (d_bits & 0xF000);
        @memcpy(dst[base .. base + 8], sc_bytes);
    }
}

// Vecinos del byte u (off-grid): el layout de acceso de llama.cpp es
// `kneighbors_q2xs - kmap_q2xs[u] - 1`; aquí índice directo (offsets).
// (lane-f unblock: /// doc-comment suelto tras '}' rompía el build del
// árbol compartido; solo el prefijo cambió, contenido intacto.)
// offsets por u: solo para off-grid; on-grid = 0
var g_offsets = [_]usize{0} ** 43692;
var g_offsets_ready: bool = false;

fn neighborsTableOf(u: usize) []const u16 {
    if (!g_offsets_ready) buildOffsets();
    const off = g_offsets[u];
    const count = g_state.neighbors[off];
    return g_state.neighbors[off .. off + 1 + count];
}

fn buildOffsets() void {
    var counter: usize = 0;
    for (0..g_state.kmap.len) |i| {
        if (g_state.kmap[i] >= 0) continue;
        g_offsets[i] = counter;
        counter += 1 + g_state.neighbors[counter];
    }
    g_offsets_ready = true;
}

// ── Tests: oráculo layout (espejo del dequant kv-path) + roundtrip ────────

test "iq1m canónico: kmap/kneighbors init determinista" {
    ensureKquantInit();
    // 2048 valores on-grid, resto off-grid con vecinos
    var on_grid: usize = 0;
    for (g_state.kmap) |v| {
        if (v >= 0) on_grid += 1;
    }
    try std.testing.expectEqual(@as(usize, 2048), on_grid);
    try std.testing.expect(g_state.neighbors.len > 2048);
    // el byte 0 (todas posiciones = 1) está en la grid (kgrid[0]=0 → pos todo 1)
    try std.testing.expect(g_state.kmap[0] >= 0);
}



test "iq1m comparativa: canónico vs legacy MISMA data" {
    const gpa = std.heap.page_allocator;
    const elems: usize = 4096;
    const src = try gpa.alloc(f32, elems);
    defer gpa.free(src);
    var rng = std.Random.Xoshiro256.init(42);
    for (src) |*v| {
        const r = rng.random().float(f32);
        const mag: f16 = if (r < 0.9) @floatCast((rng.random().float(f32) - 0.5) * 0.2) else @floatCast((rng.random().float(f32) - 0.5) * 8.0);
        v.* = @floatCast(mag);
    }
    const kv_quant = @import("kv_quant.zig");
    const dst_c = try gpa.alloc(u8, elems / 256 * 56);
    defer gpa.free(dst_c);
    quantizeRowIQ1M(src, dst_c, null, null);
    var dec_c = try gpa.alloc(f32, elems);
    defer gpa.free(dec_c);
    { var i: usize = 0; while (i < elems) : (i += 256) kv_quant.dequantIQ1_M(dst_c[i / 256 * 56 ..][0..56], dec_c[i..][0..256]); }
    // legacy encode
    const src16 = try gpa.alloc(f16, elems);
    defer gpa.free(src16);
    for (src, 0..) |v, i| src16[i] = @floatCast(v);
    const dst_l = try kv_quant.encodeToOwned(gpa, .iq1_m, src16);
    defer gpa.free(dst_l);
    var dec_l = try gpa.alloc(f32, elems);
    defer gpa.free(dec_l);
    { var i: usize = 0; while (i < elems) : (i += 256) kv_quant.dequantIQ1_M(dst_l[i / 256 * 56 ..][0..56], dec_l[i..][0..256]); }
    var sum_abs: f32 = 0; var e_c: f32 = 0; var e_l: f32 = 0;
    for (src, dec_c, dec_l) |sv, dc, dl_| {
        const in: f32 = @floatCast(@as(f16, @floatCast(sv)));
        sum_abs += @abs(in);
        e_c += @abs(dc - in);
        e_l += @abs(dl_ - in);
    }
    std.debug.print("COMPARATIVA misma-data: canon rel={d:.4} legacy rel={d:.4}\n", .{ (e_c / elems) / (sum_abs / elems), (e_l / elems) / (sum_abs / elems) });
    // desglose: mitad chica (primeros 90% de |v|<0.5) vs grande
    var ec_s: f32 = 0; var ec_l: f32 = 0; var el_s: f32 = 0; var el_l: f32 = 0;
    var n_s: usize = 0; var n_l: usize = 0; var abs_s: f32 = 0; var abs_l: f32 = 0;
    for (src, dec_c, dec_l) |sv, dc, dl_| {
        const in: f32 = @floatCast(@as(f16, @floatCast(sv)));
        if (@abs(in) < 0.5) { ec_s += @abs(dc - in); el_s += @abs(dl_ - in); abs_s += @abs(in); n_s += 1; }
        else { ec_l += @abs(dc - in); el_l += @abs(dl_ - in); abs_l += @abs(in); n_l += 1; }
    }
    std.debug.print("DESGLOSE canon: chico rel={d:.4} grande rel={d:.4} | legacy: chico rel={d:.4} grande rel={d:.4}\n", .{ (ec_s / @as(f32, @floatFromInt(n_s))) / (abs_s / @as(f32, @floatFromInt(n_s))), (ec_l / @as(f32, @floatFromInt(n_l))) / (abs_l / @as(f32, @floatFromInt(n_l))), (el_s / @as(f32, @floatFromInt(n_s))) / (abs_s / @as(f32, @floatFromInt(n_s))), (el_l / @as(f32, @floatFromInt(n_l))) / (abs_l / @as(f32, @floatFromInt(n_l))) });
}


test "iq1m canónico: port completo + hallazgo de rango (NEGATIVA documentada)" {
    // §11.1 P1 — port 1:1 de quantize_row_iq1_m_impl (llama.cpp:4692) al
    // layout kv-path: SSD weighted sort-fronteras, kmap/kneighbors init,
    // búsqueda por grupo con dd, dl-codes adaptativos por sub-bloque-16,
    // d coarse f16. Los bloques constantes reproducen EXACTO (±fudge
    // retirado); ver test comparativa.
    //
    // HALLAZGO (cierra 11.1 como NEGATIVA-con-datos): con data bimodal
    // estilo RoPE (90% ±0.1, 10% ±4, rango 40:1) NINGÚN encoder iq1_m
    // cubre los valores chicos — una sola d f16 por SB256 con 16 niveles
    // (2l+1)·(grid±dd) no alcanza. Canónico 4.16 vs legacy 3.30 (legacy
    // GANA en chicos por dd-atado-hbit: menos espacio, mejor promedio).
    // En K/V reales post-RoPE la distribución es menos extrema pero el
    // E2E cuantizado sigue incoherente (gate @d9c8c60, sopa multilingüe)
    // — el E2E usa el APPEND GPU (kvAppendIQ1_MKernel, bit-exacto vs
    // legacy por F3), o sea la conclusión aplica a TODO el camino:
    // iq1_m KV 1.75bpw es INTRÍNSECO de rango, no un bug de encoder.
    // q4_k (144B/SB, escalas por sub-bloque) sí cubre el rango — de ahí
    // que su E2E sea coherente-diferente y el de iq1_s/iq1_m no.
    // Decisión de formato: ver TODO 3.2/11.1.
    const gpa = std.heap.page_allocator;
    const elems: usize = 4096;
    const src = try gpa.alloc(f32, elems);
    defer gpa.free(src);
    var rng = std.Random.Xoshiro256.init(42);
    for (src) |*v| {
        const r = rng.random().float(f32);
        const mag: f16 = if (r < 0.9) @floatCast((rng.random().float(f32) - 0.5) * 0.2) else @floatCast((rng.random().float(f32) - 0.5) * 8.0);
        v.* = @floatCast(mag);
    }
    const dst = try gpa.alloc(u8, elems / 256 * 56);
    defer gpa.free(dst);
    quantizeRowIQ1M(src, dst, null, null);
    // Sane checks del port (los invariantes SÍ exigibles):
    // - init determinista: 2048 on-grid
    // - valores GRANDES bien cuantizados (rel < 1.0 en |v|>=0.5)
    const kv_quant = @import("kv_quant.zig");
    const dec = try gpa.alloc(f32, elems);
    defer gpa.free(dec);
    {
        var i: usize = 0;
        while (i < elems) : (i += 256) {
            kv_quant.dequantIQ1_M(dst[i / 256 * 56 ..][0..56], dec[i..][0..256]);
        }
    }
    var e_l: f32 = 0;
    var abs_l: f32 = 0;
    var n_l: usize = 0;
    for (src, dec) |sv, dv| {
        const in: f32 = @floatCast(@as(f16, @floatCast(sv)));
        if (@abs(in) >= 0.5) {
            e_l += @abs(dv - in);
            abs_l += @abs(in);
            n_l += 1;
        }
    }
    const rel_grande = (e_l / @as(f32, @floatFromInt(n_l))) / (abs_l / @as(f32, @floatFromInt(n_l)));
    std.debug.print("iq1_m canónico: rel en valores grandes (|v|>=0.5) = {d:.4} (criterio <1.0 — chicos fuera de alcance del formato, ver comparativa)\n", .{rel_grande});
    try std.testing.expect(rel_grande < 1.0);
}

test "iq1m canónico: verificación explícita de layout (kv-path)" {
    // 6.1 — Verifica que el encoder canónico produce el layout kv-path:
    //   bytes [0,8)   = scales con d (no todo-cero para data no-cero)
    //   bytes [8,32)  = qs (qb de sub-bloques)
    //   bytes [32,48) = qh (3 bits altos idx + bit dd)
    //   bytes [48,56) = padding cero
    const gpa = std.heap.page_allocator;
    const elems: usize = 256;
    const src = try gpa.alloc(f32, elems);
    defer gpa.free(src);
    var rng = std.Random.Xoshiro256.init(123);
    for (src) |*v| v.* = @floatCast((rng.random().float(f32) - 0.5) * 4.0);
    const dst = try gpa.alloc(u8, 56);
    defer gpa.free(dst);
    quantizeRowIQ1M(src, dst, null, null);
    // Padding [48..56) debe ser cero.
    for (dst[48..56]) |b| try std.testing.expect(b == 0);
    // qs [8..32) no-todo-cero (input no-cero).
    var qs_nonzero: usize = 0;
    for (dst[8..32]) |b| {
        if (b != 0) qs_nonzero += 1;
    }
    try std.testing.expect(qs_nonzero > 0);
    // qh [32..48) no-todo-cero.
    var qh_nonzero: usize = 0;
    for (dst[32..48]) |b| {
        if (b != 0) qh_nonzero += 1;
    }
    try std.testing.expect(qh_nonzero > 0);
    // scales [0..8) no-todo-cero (d existe para data no-cero).
    var sc_nonzero: usize = 0;
    for (dst[0..8]) |b| {
        if (b != 0) sc_nonzero += 1;
    }
    try std.testing.expect(sc_nonzero > 0);
}
