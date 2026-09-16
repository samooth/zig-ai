const std = @import("std");
const cudaz = @import("cudaz");
pub const fa_config = @import("fa_config.zig");
pub const fa_utils = @import("fa_utils.zig");
const fa_kernels = @import("fa_kernels.zig");
const Tensor = @import("core").Tensor;
const debugz = @import("debug");

const FlashAttentionConfig = fa_config.FlashAttentionConfig;
const DType = fa_config.DType;
const CudaKernel = fa_kernels.CudaKernel;
const FlashAttentionBuffers = fa_kernels.FlashAttentionBuffers;

pub const FlashAttentionError = error{
    CudaError,
    CudaOutOfMemory,
    CudaInvalidDevice,
    CudaInvalidValue,
    CudaLaunchFailed,
    CudaUnknown,
    InvalidConfig,
    UnsupportedDtype,
    KernelLaunchFailed,
    MemoryAllocationFailed,
    PtxNotFound,
    CpuNTooLarge,
};

/// Motor principal de FlashAttention
pub const FlashAttention = struct {
    allocator: std.mem.Allocator,
    config: FlashAttentionConfig,
    kernel: CudaKernel,
    buffers: FlashAttentionBuffers,
    stream: cudaz.CUstream,
    h_q: []u8,
    h_k: []u8,
    h_v: []u8,
    h_o: []u8,
    h_q_raw: *anyopaque,
    h_k_raw: *anyopaque,
    h_v_raw: *anyopaque,
    h_o_raw: *anyopaque,
    cuda_context: cudaz.CUcontext,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: FlashAttentionConfig, ptx_path: []const u8) !Self {
        try config.validate();
        try cudaz.cuInit(0);
        const device = try cudaz.cuDeviceGet(0);
        const ctx = try cudaz.cuCtxCreate(0, device);
        var kernel = try CudaKernel.load(ptx_path, "launch_flash_attention");
        errdefer kernel.unload();
        const stream = try cudaz.cuStreamCreate(0);
        var buffers = try FlashAttentionBuffers.alloc(config);
        errdefer buffers.free();
        const total_bytes = config.total_qkv_bytes();
        const h_q_raw = try cudaz.cuMemAllocHost(total_bytes);
        const h_k_raw = try cudaz.cuMemAllocHost(total_bytes);
        const h_v_raw = try cudaz.cuMemAllocHost(total_bytes);
        const h_o_raw = try cudaz.cuMemAllocHost(total_bytes);
        return .{
            .allocator = allocator,
            .config = config,
            .kernel = kernel,
            .buffers = buffers,
            .stream = stream,
            .cuda_context = ctx,
            .h_q_raw = h_q_raw,
            .h_k_raw = h_k_raw,
            .h_v_raw = h_v_raw,
            .h_o_raw = h_o_raw,
            .h_q = @as([*]u8, @ptrCast(h_q_raw))[0..total_bytes],
            .h_k = @as([*]u8, @ptrCast(h_k_raw))[0..total_bytes],
            .h_v = @as([*]u8, @ptrCast(h_v_raw))[0..total_bytes],
            .h_o = @as([*]u8, @ptrCast(h_o_raw))[0..total_bytes],
        };
    }

    pub fn deinit(self: *Self) void {
        cudaz.cuMemFreeHost(self.h_q_raw);
        cudaz.cuMemFreeHost(self.h_k_raw);
        cudaz.cuMemFreeHost(self.h_v_raw);
        cudaz.cuMemFreeHost(self.h_o_raw);
        self.buffers.free();
        cudaz.cuStreamDestroy(self.stream);
        self.kernel.unload();
        cudaz.cuCtxDestroy(self.cuda_context);
    }

    /// Forward: Q, K, V host -> GPU -> FA -> host
    pub fn forward(self: *Self, Q: Tensor(f16), K: Tensor(f16), V: Tensor(f16), O: *Tensor(f16)) !void {
        const cfg = self.config;
        const total_bytes = cfg.total_qkv_bytes();
        if (Q.shape.len != 4 or K.shape.len != 4 or V.shape.len != 4 or O.shape.len != 4)
            return FlashAttentionError.InvalidConfig;
        if (Q.shape[0] != cfg.batch_size or Q.shape[1] != cfg.num_heads or Q.shape[2] != cfg.N or Q.shape[3] != cfg.d)
            return FlashAttentionError.InvalidConfig;

        @memcpy(self.h_q[0..total_bytes], std.mem.sliceAsBytes(Q.data));
        @memcpy(self.h_k[0..total_bytes], std.mem.sliceAsBytes(K.data));
        @memcpy(self.h_v[0..total_bytes], std.mem.sliceAsBytes(V.data));

        try cudaz.cuMemcpyHtoDAsync(self.buffers.d_q, @intFromPtr(self.h_q.ptr), total_bytes, self.stream);
        try cudaz.cuMemcpyHtoDAsync(self.buffers.d_k, @intFromPtr(self.h_k.ptr), total_bytes, self.stream);
        try cudaz.cuMemcpyHtoDAsync(self.buffers.d_v, @intFromPtr(self.h_v.ptr), total_bytes, self.stream);

        try fa_kernels.launchFlashAttentionV1(self.kernel.function, cfg, self.buffers, self.stream);

        try cudaz.cuMemcpyDtoHAsync(@intFromPtr(self.h_o.ptr), self.buffers.d_o, total_bytes, self.stream);
        try cudaz.cuStreamSynchronize(self.stream);
        @memcpy(std.mem.sliceAsBytes(O.data), self.h_o[0..total_bytes]);
    }

    /// Forward con secuencia runtime T (7.1c): los buffers/device están
    /// dimensionados a N (capacidad), pero el launch corre con T tokens
    /// reales — el kernel recibe T como argumento. Copia solo T×heads×d.
    /// Los tensors Q/K/V/O deben ser views head-major [b, h, T, d] sobre
    /// los slabs de capacidad N (T ≤ N).
    pub fn forwardSeq(self: *Self, Q: Tensor(f16), K: Tensor(f16), V: Tensor(f16), O: *Tensor(f16), t_seq: usize) !void {
        const cfg = self.config;
        if (t_seq == 0 or t_seq > cfg.N) return FlashAttentionError.InvalidConfig;
        if (Q.shape.len != 4 or K.shape.len != 4 or V.shape.len != 4 or O.shape.len != 4)
            return FlashAttentionError.InvalidConfig;
        if (Q.shape[0] != cfg.batch_size or Q.shape[1] != cfg.num_heads or Q.shape[2] != t_seq or Q.shape[3] != cfg.d)
            return FlashAttentionError.InvalidConfig;

        const seq_bytes = cfg.batch_size * cfg.num_heads * t_seq * cfg.d * cfg.dtype.size();
        @memcpy(self.h_q[0..seq_bytes], std.mem.sliceAsBytes(Q.data));
        @memcpy(self.h_k[0..seq_bytes], std.mem.sliceAsBytes(K.data));
        @memcpy(self.h_v[0..seq_bytes], std.mem.sliceAsBytes(V.data));

        try cudaz.cuMemcpyHtoDAsync(self.buffers.d_q, @intFromPtr(self.h_q.ptr), seq_bytes, self.stream);
        try cudaz.cuMemcpyHtoDAsync(self.buffers.d_k, @intFromPtr(self.h_k.ptr), seq_bytes, self.stream);
        try cudaz.cuMemcpyHtoDAsync(self.buffers.d_v, @intFromPtr(self.h_v.ptr), seq_bytes, self.stream);

        var cfg_seq = cfg;
        cfg_seq.N = t_seq;
        try fa_kernels.launchFlashAttentionV1(self.kernel.function, cfg_seq, self.buffers, self.stream);

        try cudaz.cuMemcpyDtoHAsync(@intFromPtr(self.h_o.ptr), self.buffers.d_o, seq_bytes, self.stream);
        try cudaz.cuStreamSynchronize(self.stream);
        @memcpy(std.mem.sliceAsBytes(O.data), self.h_o[0..seq_bytes]);
    }

    /// Forward device-to-device (todo en GPU)
    pub fn forwardDevice(self: *Self, d_q: cudaz.CUdeviceptr, d_k: cudaz.CUdeviceptr, d_v: cudaz.CUdeviceptr, d_o: cudaz.CUdeviceptr) !void {
        const temp = FlashAttentionBuffers{
            .d_q = d_q,
            .d_k = d_k,
            .d_v = d_v,
            .d_o = d_o,
            .bytes = self.config.total_qkv_bytes(),
        };
        try fa_kernels.launchFlashAttentionV1(self.kernel.function, self.config, temp, self.stream);
        try cudaz.cuStreamSynchronize(self.stream);
    }

    /// Forward batch
    pub fn forwardBatch(self: *Self, Q: []const Tensor(f16), K: []const Tensor(f16), V: []const Tensor(f16), O: []*Tensor(f16)) !void {
        if (Q.len != K.len or Q.len != V.len or Q.len != O.len) return FlashAttentionError.InvalidConfig;
        for (Q, K, V, O) |q, k, v, *o| try self.forward(q, k, v, o);
    }

    pub fn printDeviceInfo() !void {
        try cudaz.cuInit(0);
        const device = try cudaz.cuDeviceGet(0);
        var allocator = std.heap.page_allocator;
        const name = try cudaz.getDeviceName(device, allocator);
        defer allocator.free(name);
        const mem = try cudaz.getDeviceTotalMem(device);
        const mem_str = try fa_utils.formatBytes(allocator, mem);
        defer allocator.free(mem_str);
        debugz.dbg.print("CUDA Device: {s}\n", .{name});
        debugz.dbg.print("Total Memory: {s}\n", .{mem_str});
    }
};

/// Version CPU para validacion
pub const FlashAttentionCpu = struct {
    allocator: std.mem.Allocator,
    config: FlashAttentionConfig,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: FlashAttentionConfig) Self {
        return .{ .allocator = allocator, .config = config };
    }

    /// Forward CPU stride-aware + GQA (7.2, lane-f).
    ///
    /// Tensors 4D [b, h, T, d]: Q con num_heads (q) y T_q tokens; K/V con
    /// num_kv_heads y T_kv tokens (T_kv >= T_q). Se respetan strides y
    /// offset (los callers pasan views sobre slabs de capacidad mayor).
    /// La máscara causal es [T_q x T_kv]: token q_i atiende posiciones
    /// kv_j con j <= offset_kv + i.
    ///
    /// La GQA se resuelve con el mapeo estándar q_h -> kv_h = q_h % n_kv
    /// (llama.cpp ggml_repeat — mismo criterio que el fix ΔNet 7.1b).
    pub fn forward(self: Self, Q: Tensor(f16), K: Tensor(f16), V: Tensor(f16), O: *Tensor(f16)) !void {
        const cfg = self.config;
        const batch_size = Q.shape[0];
        const n_q_heads = Q.shape[1];
        const t_q = Q.shape[2];
        const d = Q.shape[3];
        const n_kv_heads = K.shape[1];
        const t_kv = K.shape[2];
        const scale = cfg.scale();
        const causal = cfg.causal;

        std.debug.assert(V.shape[1] == n_kv_heads and V.shape[2] == t_kv);
        std.debug.assert(O.shape[1] == n_q_heads and O.shape[2] == t_q);
        std.debug.assert(K.shape[3] == d and V.shape[3] == d and O.shape[3] == d);
        if (n_q_heads % n_kv_heads != 0) return FlashAttentionError.InvalidConfig;

        // Offset kv: el primer token de Q es el token global
        // (t_kv - t_q) — en decode T_q=1 y t_kv=historia completa.
        const kv_offset = t_kv - t_q;

        // Puntero a la fila [b,h,t,d] de un tensor 4D respetando strides
        const rowOf = struct {
            fn go(t: Tensor(f16), b: usize, h: usize, tpos: usize) [*]f16 {
                const off = t.offset +
                    b * t.strides[0] +
                    h * t.strides[1] +
                    tpos * t.strides[2];
                return t.data.ptr + off;
            }
        }.go;

        const scores = try self.allocator.alloc(f32, t_q * t_kv);
        defer self.allocator.free(scores);

        for (0..batch_size) |b| {
            for (0..n_q_heads) |qh| {
                const kv_h = qh % n_kv_heads; // GQA mapping (ggml_repeat)
                for (0..t_q) |i| {
                    const q_row = rowOf(Q, b, qh, i);
                    // Scores fila i: dot(Q_i, K_j) para j <= kv_offset+i
                    for (0..t_kv) |j| {
                        var dot: f32 = 0;
                        const k_row = rowOf(K, b, kv_h, j);
                        for (0..d) |k| {
                            dot += fa_utils.f16ToF32(q_row[k]) * fa_utils.f16ToF32(k_row[k]);
                        }
                        scores[i * t_kv + j] = if (causal and j > kv_offset + i)
                            -std.math.inf(f32)
                        else
                            dot * scale;
                    }
                    // Softmax fila i (numericalemente estable)
                    var max_val: f32 = -std.math.inf(f32);
                    for (0..t_kv) |j| {
                        if (scores[i * t_kv + j] > max_val) max_val = scores[i * t_kv + j];
                    }
                    var sum: f32 = 0;
                    for (0..t_kv) |j| {
                        scores[i * t_kv + j] = @exp(scores[i * t_kv + j] - max_val);
                        sum += scores[i * t_kv + j];
                    }
                    // Salida: suma ponderada V (acumulador f32 — evitar
                    // error de redondeo acumulado en f16)
                    const o_row = rowOf(O.*, b, qh, i);
                    const acc = try self.allocator.alloc(f32, d);
                    defer self.allocator.free(acc);
                    @memset(acc, 0);
                    for (0..t_kv) |j| {
                        const w = scores[i * t_kv + j];
                        if (w == 0) continue;
                        const v_row = rowOf(V, b, kv_h, j);
                        for (0..d) |k| {
                            acc[k] += w * fa_utils.f16ToF32(v_row[k]);
                        }
                    }
                    if (sum != 0) {
                        const inv_sum = 1.0 / sum;
                        for (0..d) |k| o_row[k] = fa_utils.f32ToF16(acc[k] * inv_sum);
                    } else {
                        for (0..d) |k| o_row[k] = fa_utils.f32ToF16(acc[k]);
                    }
                }
            }
        }
    }
};
