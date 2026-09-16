//! ClipAttention — self-attention del ViT con QKV fused y M-RoPE VISION.
//!
//! Referencias (llama.cpp):
//!   - Grafo Qwen3-VL: models/qwen3vl.cpp:79-117 (QKV fused → split por
//!     offsets 0 / n_embd / 2·n_embd → MRoPE vision → build_attn con
//!     kq_scale=1/sqrt(d_head) → out proj)
//!   - build_attn: clip-graph.h:120-127 (softmax estándar, sin causal mask)
//!
//! Layout interno:
//!   x:       [n_pos, n_embd] row-major (tokens × features)
//!   Q/K/V:   [n_pos, n_head, head_dim] (pos-major; el layout que consume
//!            applyMRopeVision)
//!   scores:  [n_head, n_pos, n_pos]
const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const QuantWeight = @import("quant_weight").QuantWeight;
const mrope = @import("mrope_vision");
const debugz = @import("debug");
const timez = @import("time");
// Post-freeze 2026-09-13: pool softmax por presupuesto central, no
// getCpuCount() (lógicos) — el ViT corría en el mismo host que matmul.
const resources = @import("resources");

pub const ClipAttentionError = error{
    MissingWeights,
    ShapeMismatch,
    OutOfMemory,
};

pub const ClipAttention = struct {
    allocator: std.mem.Allocator,
    engine: *matmul.MatmulEngine,

    // Pesos dequantizados una vez en init (los del ViT son pequeños:
    // [3n_embd, n_embd] f16/f32)
    qkv_w_t: Tensor(f32), // [3*n_embd, n_embd] transpuesto (linearProjection)
    qkv_b: ?[]f32, // [3*n_embd]
    o_w_t: Tensor(f32), // [n_embd, n_embd] transpuesto
    o_b: ?[]f32, // [n_embd]

    n_embd: usize,
    n_head: usize,
    head_dim: usize,
    n_head_kv: usize,

    const Self = @This();

    /// Carga y dequantiza los pesos desde el mmproj.
    /// `qkv_w`/`o_w` son [in, out] GGUF → transponer a [out, in].
    pub fn init(
        allocator: std.mem.Allocator,
        engine: *matmul.MatmulEngine,
        m: anytype, // *const MmprojModel (anytype para evitar ciclo de imports)
        il: usize,
        n_embd: usize,
        n_head: usize,
        head_dim: usize,
        n_head_kv: usize,
    ) !Self {
        const qkv_w = m.blk(il, "attn_qkv.weight") orelse return ClipAttentionError.MissingWeights;
        const o_w = m.blk(il, "attn_out.weight") orelse return ClipAttentionError.MissingWeights;

        var qkv_w_t = try m.dequantToF32Transposed(qkv_w, 3 * n_embd, n_embd);
        errdefer qkv_w_t.deinit();
        var o_w_t = try m.dequantToF32Transposed(o_w, n_embd, n_embd);
        errdefer o_w_t.deinit();

        var qkv_b: ?[]f32 = null;
        errdefer if (qkv_b) |b| allocator.free(b);
        if (m.blk(il, "attn_qkv.bias")) |w| {
            qkv_b = try dequantF32Slice(allocator, w);
        }
        var o_b: ?[]f32 = null;
        errdefer if (o_b) |b| allocator.free(b);
        if (m.blk(il, "attn_out.bias")) |w| {
            o_b = try dequantF32Slice(allocator, w);
        }

        return .{
            .allocator = allocator,
            .engine = engine,
            .qkv_w_t = qkv_w_t,
            .qkv_b = qkv_b,
            .o_w_t = o_w_t,
            .o_b = o_b,
            .n_embd = n_embd,
            .n_head = n_head,
            .head_dim = head_dim,
            .n_head_kv = n_head_kv,
        };
    }

    pub fn deinit(self: *Self) void {
        self.qkv_w_t.deinit();
        self.o_w_t.deinit();
        if (self.qkv_b) |b| self.allocator.free(b);
        if (self.o_b) |b| self.allocator.free(b);
    }

    /// Forward: x [n_pos, n_embd] → out [n_pos, n_embd].
    /// `ids`: position ids 4D por token (visionPosIds).
    /// `scratch`: >= n_pos·3·n_embd + n_pos·n_embd + n_head·n_pos² + n_pos·head_dim
    /// Requisito de scratch para forward(n_pos): qkv + Q/K/V split +
    /// scores [n_head·n_pos²] + rope cache.
    pub fn scratchNeed(self: *const Self, n_pos: usize) usize {
        return n_pos * 3 * self.n_embd + // qkv buf
            3 * n_pos * self.n_embd + // Q/K/V split
            self.n_head * n_pos * n_pos + // scores
            n_pos * self.head_dim + // rope cache
            2 * self.n_head * n_pos * self.head_dim; // pack Q/K (GEMM por head)
    }

    pub fn forward(
        self: *const Self,
        x: []const f32, // [n_pos, n_embd]
        out: []f32, // [n_pos, n_embd]
        n_pos: usize,
        ids: []const [4]i32,
        scratch: []f32,
    ) !void {
        const n_embd = self.n_embd;
        const n_head = self.n_head;
        const head_dim = self.head_dim;
        const n_head_kv = self.n_head_kv;

        // ── 1. QKV proyección: [n_pos, n_embd] → [n_pos, 3·n_embd]
        const qkv_sz = n_pos * 3 * n_embd;
        if (scratch.len < self.scratchNeed(n_pos))
            return ClipAttentionError.ShapeMismatch;
        const qkv = scratch[0..qkv_sz];

        const t_att = timez.Timer.start();
        const x_shape = [_]usize{ n_pos, n_embd };
        var x_strides = [_]usize{ n_embd, 1 };
        const x_t = Tensor(f32){
            .data = @constCast(x),
            .shape = &x_shape,
            .strides = &x_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        const qkv_shape = [_]usize{ n_pos, 3 * n_embd };
        var qkv_strides = [_]usize{ 3 * n_embd, 1 };
        var qkv_t = Tensor(f32){
            .data = qkv,
            .shape = &qkv_shape,
            .strides = &qkv_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        try self.engine.linearProjection(f32, x_t, self.qkv_w_t, &qkv_t);

        if (self.qkv_b) |b| {
            for (0..n_pos) |t| {
                const row = qkv[t * 3 * n_embd ..][0 .. 3 * n_embd];
                for (b, 0..) |bv, i| row[i] += bv;
            }
        }

        if (debugz.dbg.perf_mm) {
            debugz.dbg.printLevel(.info, "[mmproj-perf] attn qkv proj: {d} ms\n", .{@divTrunc(t_att.read(), std.time.ns_per_ms)});
        }
        // ── 2. Split Q/K/V: offsets 0, n_embd, 2·n_embd (qwen3vl.cpp:84-97)
        //    → layout [n_pos, n_head, head_dim]
        const q_buf = scratch[qkv_sz .. qkv_sz + n_pos * n_embd];
        const k_buf = scratch[qkv_sz + n_pos * n_embd .. qkv_sz + 2 * n_pos * n_embd];
        const v_buf = scratch[qkv_sz + 2 * n_pos * n_embd .. qkv_sz + 3 * n_pos * n_embd];

        const kv_stride = n_head_kv * head_dim; // ≤ n_embd (GQA)
        for (0..n_pos) |t| {
            const row = qkv[t * 3 * n_embd ..][0 .. 3 * n_embd];
            // Q: [n_head·head_dim] desde offset 0
            @memcpy(q_buf[t * n_embd ..][0..n_embd], row[0..n_embd]);
            // K: [n_head_kv·head_dim] desde offset n_embd
            @memcpy(k_buf[t * kv_stride ..][0..kv_stride], row[n_embd .. n_embd + kv_stride]);
            // V: [n_head_kv·head_dim] desde offset 2·n_embd... PERO ggml
            // pone K y V contiguos: Q [n_embd] | K [n_embd] | V [n_embd]
            // (3·n_embd siempre). Con GQA real (n_head_kv < n_head) el GGUF
            // tiene qkv rows = n_embd + 2·kv_stride... El ViT de Qwen3-VL
            // usa n_head_kv == n_head (sin GQA), así que aquí asumimos
            // 3·n_embd rows simétrico y kv_stride == n_embd.
            if (n_head_kv != n_head) return ClipAttentionError.ShapeMismatch;
            @memcpy(v_buf[t * n_embd ..][0..n_embd], row[2 * n_embd .. 3 * n_embd]);
        }

        // ── 3. M-RoPE VISION sobre Q/K (sections = [hd/4 ×4], base 10000)
        const sections = [4]usize{
            head_dim / 4,
            head_dim / 4,
            head_dim / 4,
            head_dim / 4,
        };
        const rope_scratch = scratch[qkv_sz + 3 * n_pos * n_embd ..][0 .. n_pos * head_dim];
        mrope.applyMRopeVision(f32, q_buf, k_buf, ids, n_head, head_dim, sections, 10000.0, rope_scratch);

        // ── 4. Attention scores por head: scores[h] = Q·K^T / sqrt(d_head)
        // PERF: GEMM por head vía engine.gemm (paralelizado/vectorizado)
        // — el bucle naive O(n²·hd) scalar era el 85% del encode (1.75s/capa
        // a n=1200). Layout Q/K: [n_pos, n_head, hd] → por head h,
        // Q_h [n_pos, hd] con stride n_embd entre filas: usamos views con
        // strides — gemm exige contiguos, así que empaquetamos por head en
        // scratch aux (pack_q/pack_k reutilizan el q_buf/k_buf O(1) copia).
        const scores = scratch[qkv_sz + 3 * n_pos * n_embd + n_pos * head_dim ..][0 .. n_head * n_pos * n_pos];
        const kq_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        // pack buffers: [n_head][n_pos*hd] contiguos por head
        const pack_off = qkv_sz + 3 * n_pos * n_embd + n_pos * head_dim + n_head * n_pos * n_pos;
        const pack_q = scratch[pack_off..][0 .. n_head * n_pos * head_dim];
        const pack_k = scratch[pack_off + n_head * n_pos * head_dim ..][0 .. n_head * n_pos * head_dim];
        for (0..n_head) |h| {
            // pack Q_h / K_h [n_pos, hd] (extraer de [n_pos, n_embd] intercalado)
            for (0..n_pos) |t| {
                @memcpy(
                    pack_q[h * n_pos * head_dim + t * head_dim ..][0..head_dim],
                    q_buf[t * n_embd + h * head_dim ..][0..head_dim],
                );
                @memcpy(
                    pack_k[h * n_pos * head_dim + t * head_dim ..][0..head_dim],
                    k_buf[t * n_embd + h * head_dim ..][0..head_dim],
                );
            }
        }
        var qh_shape = [2]usize{ n_pos, head_dim };
        var qh_strides = [2]usize{ head_dim, 1 };
        var kh_shape = [2]usize{ n_pos, head_dim };
        var kh_strides = [2]usize{ head_dim, 1 };
        var sc_shape = [2]usize{ n_pos, n_pos };
        var sc_strides = [2]usize{ n_pos, 1 };
        if (debugz.dbg.dump_mm_input and debugz.dbg.at(.info)) {
            debugz.dbg.printLevel(.info, "[mmproj-qkv] x[0..3]={d:.4},{d:.4},{d:.4} qkv[t0][0..3]={d:.4},{d:.4},{d:.4} qkv[t1][0..3]={d:.4}\n", .{
                x[0],            x[1],   x[2],
                qkv[0],          qkv[1], qkv[2],
                qkv[3 * n_embd],
            });
            debugz.dbg.printLevel(.info, "[mmproj-rope] qh0p0[3]={d:.4},{d:.4},{d:.4} last3={d:.4},{d:.4},{d:.4}\n", .{
                q_buf[0],  q_buf[1],  q_buf[2],
                q_buf[61], q_buf[62], q_buf[63],
            });
            debugz.dbg.printLevel(.info, "[mmproj-rope-p1] qh0[3]={d:.4},{d:.4},{d:.4}\n", .{
                q_buf[n_embd], q_buf[n_embd + 1], q_buf[n_embd + 2],
            });
        }
        // Fase A: gemm de scores por head (cada gemm paraleliza con el pool
        // del engine — secuencial entre heads). kq_scale dentro del softmax.
        for (0..n_head) |h| {
            const Q_h = Tensor(f32){
                .data = pack_q[h * n_pos * head_dim ..][0 .. n_pos * head_dim],
                .shape = &qh_shape,
                .strides = &qh_strides,
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };
            const K_h = Tensor(f32){
                .data = pack_k[h * n_pos * head_dim ..][0 .. n_pos * head_dim],
                .shape = &kh_shape,
                .strides = &kh_strides,
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };
            var S_h = Tensor(f32){
                .data = scores[h * n_pos * n_pos ..][0 .. n_pos * n_pos],
                .shape = &sc_shape,
                .strides = &sc_strides,
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };
            // S = Q_h · K_h^T (trans_b=true — path SIMD sin transpose-alloc)
            try self.engine.gemm(f32, Q_h, K_h, &S_h, false, true);
        }

        // Fase B: kq_scale + softmax in-place, paralelo por filas
        // (n_head·n_pos filas × n_pos elems; 23M exps en ViT 1200 — el bucle
        // scalar era ~400ms/capa). Threads por rangos, patrón parallel.zig.
        {
            const total_rows = n_head * n_pos;
            // Presupuesto central (2026-09-13): físico−load, ≤4 en tests.
            var fba_buf: [256]u8 = undefined;
            var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
            const budget = resources.computeThreadBudget(fba.allocator(), "clip_softmax");
            const n_threads = @min(@max(1, budget), total_rows);
            const rows_per = total_rows / n_threads;
            const rem = total_rows % n_threads;
            var threads: [64]std.Thread = undefined;
            var spawned: usize = 0;
            var r0: usize = 0;
            for (0..n_threads) |ti| {
                const extra: usize = if (ti < rem) 1 else 0;
                const r1 = r0 + rows_per + extra;
                if (r1 > r0 and spawned < threads.len) {
                    threads[spawned] = std.Thread.spawn(.{}, softmaxRows, .{
                        scores, r0, r1, n_pos, kq_scale,
                    }) catch {
                        softmaxRows(scores, r0, r1, n_pos, kq_scale);
                        r0 = r1;
                        continue;
                    };
                    spawned += 1;
                }
                r0 = r1;
            }
            for (threads[0..spawned]) |th| th.join();
        }

        if (debugz.dbg.perf_mm) {
            debugz.dbg.printLevel(.info, "[mmproj-perf] attn rope+scores: {d} ms\n", .{@divTrunc(t_att.read(), std.time.ns_per_ms)});
        }
        // ── 5. Attn out: out[h] = S_h · V_h. V se empaqueta TRANSPUESTO
        // [hd, n_pos] y se usa gemm(trans_b=true) — el path trans_b=false
        // del backend parallel llama B.transpose() que alloca con
        // B.allocator (null en scratch views) ⇒ SIGSEGV. Con Vᵀ el gemm
        // consume el buffer contiguo directo.
        const attn_out = scratch[qkv_sz .. qkv_sz + n_pos * n_embd];
        var vt_shape = [2]usize{ head_dim, n_pos };
        var vt_strides = [2]usize{ n_pos, 1 };
        for (0..n_head) |h| {
            // pack Vᵀ_h [hd, n_pos]: (pack_k reutilizado como Vᵀ buffer)
            for (0..n_pos) |t| {
                const vsrc = v_buf[t * n_embd + h * head_dim ..][0..head_dim];
                for (0..head_dim) |d| {
                    pack_k[h * n_pos * head_dim + d * n_pos + t] = vsrc[d];
                }
            }
            const Vt_h = Tensor(f32){
                .data = pack_k[h * n_pos * head_dim ..][0 .. n_pos * head_dim],
                .shape = &vt_shape,
                .strides = &vt_strides,
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };
            const S_h = Tensor(f32){
                .data = scores[h * n_pos * n_pos ..][0 .. n_pos * n_pos],
                .shape = &sc_shape,
                .strides = &sc_strides,
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };
            var O_h = Tensor(f32){
                .data = pack_q[h * n_pos * head_dim ..][0 .. n_pos * head_dim],
                .shape = &qh_shape,
                .strides = &qh_strides,
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };
            // O = S · V = S · (Vᵀ)ᵀ ⇒ gemm(trans_b=true)
            try self.engine.gemm(f32, S_h, Vt_h, &O_h, false, true);
            // unpack O_h → attn_out [n_pos, n_embd]
            for (0..n_pos) |t| {
                @memcpy(
                    attn_out[t * n_embd + h * head_dim ..][0..head_dim],
                    pack_q[h * n_pos * head_dim + t * head_dim ..][0..head_dim],
                );
            }
        }

        // ── 6. Output projection
        const attn_shape = [_]usize{ n_pos, n_embd };
        var attn_strides = [_]usize{ n_embd, 1 };
        const attn_t = Tensor(f32){
            .data = attn_out,
            .shape = &attn_shape,
            .strides = &attn_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        const out_shape = [_]usize{ n_pos, n_embd };
        var out_strides = [_]usize{ n_embd, 1 };
        var out_t = Tensor(f32){
            .data = out,
            .shape = &out_shape,
            .strides = &out_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        try self.engine.linearProjection(f32, attn_t, self.o_w_t, &out_t);
        if (self.o_b) |b| {
            for (0..n_pos) |t| {
                const row = out[t * n_embd ..][0..n_embd];
                for (b, 0..) |bv, i| row[i] += bv;
            }
        }
    }
};

/// Softmax in-place por filas de la matriz de scores plana [total_rows × row_len],
/// con kq_scale aplicado antes de la resta del max (fusiona el escalado).
/// Filas [r0, r1). Sin máscara causal (ViT bidireccional). Thread-safe: cada
/// fila es exclusiva de un thread.
fn softmaxRows(scores: []f32, r0: usize, r1: usize, row_len: usize, kq_scale: f32) void {
    for (r0..r1) |r| {
        const row = scores[r * row_len ..][0..row_len];
        var max_v: f32 = -std.math.inf(f32);
        for (row) |v| max_v = @max(max_v, v * kq_scale);
        var sum: f32 = 0;
        for (row) |*v| {
            v.* = @exp(v.* * kq_scale - max_v);
            sum += v.*;
        }
        for (row) |*v| v.* /= sum;
    }
}

/// Dequantiza un QuantWeight 1D pequeño a slice f32 (norms, biases).
pub fn dequantF32Slice(allocator: std.mem.Allocator, w: QuantWeight) ![]f32 {
    const n: usize = @intCast(w.numel());
    const out = try allocator.alloc(f32, n);
    errdefer allocator.free(out);
    @import("gguf").dequantTensor(w.info, w.bytes, out) catch {
        allocator.free(out);
        return ClipAttentionError.MissingWeights;
    };
    return out;
}
