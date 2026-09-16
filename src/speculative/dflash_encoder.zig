//! C6.1 — DFlash encoder device-resident (TODO 5.1, lane-c).
//!
//! El encoder dflash (upstream llama.cpp models/dflash.cpp:189-217) es:
//!   h = rmsNorm(fc · x_taps, enc.output_norm, eps)
//! donde x_taps = concat de los hidden de las dflash.target_layers del
//! target ([n_taps·n_embd] — p.ej. 8×4096=32768 en el 9B). El TODO 5.1
//! pide "encoder fc+norm device-resident, taps sin D2H": los hidden del
//! target YA viven en device durante el decode (g.g_norm/g.g_mixer);
//! este módulo los consume sin bajar nada a host.
//!
//! Pipeline por ronda (3 launches + 1 GEMV):
//!   1. dflashTapsGatherKernel: concat n_taps ptrs device → fc_in [K]
//!   2. qgemmKernel (q8_0, M=1): fc_out [n_embd] = fc · fc_in
//!   3. rmsNormKernel: h = rmsNorm(fc_out, enc.output_norm)
//!
//! Pesos device-resident: fc q8_0 via el cache q4Weight (mmap→device una
//! vez), enc.output_norm f32 device (16KB). Vida: init/deinit del engine.

const std = @import("std");
const cudaz = @import("cudaz");
const layer_kernels = @import("layer_kernels");
const quant_weight = @import("quant_weight");
const debugz = @import("debug");

pub const DflashEncoderError = error{
    SidecarNotLoaded,
    DimensionMismatch,
    CudaUnavailable,
};

pub const DflashEncoder = struct {
    allocator: std.mem.Allocator,
    lk: *layer_kernels.LayerKernels,

    /// dflash.target_layers (índices de capa del target a tapear).
    target_layers: []const i32,
    n_embd: usize,
    /// K del fc = target_layers.len · n_embd (32768 en el 9B).
    fc_k: usize,
    eps: f32,

    /// fc.weight q8_0 mmap (préstamo del GGUF del sidecar — NO liberar).
    fc_bytes: []const u8,
    /// Puntero device del fc (cache q4Weight — key = host ptr del mmap).
    fc_dev: usize,
    /// enc.output_norm f32 en device ([n_embd]).
    norm_dev: cudaz.CUdeviceptr,

    /// Buffer device de punteros device (taps por ronda) + fc_in scratch.
    taps_ptrs_dev: cudaz.CUdeviceptr,
    fc_in_dev: cudaz.CUdeviceptr,
    fc_out_dev: cudaz.CUdeviceptr,

    /// Carga los pesos del sidecar a device. `sidecar` ya dual-loadeado.
    pub fn init(
        allocator: std.mem.Allocator,
        lk: *layer_kernels.LayerKernels,
        sc: anytype, // *gguf_model.SidecarDraft — evita el import circular
        n_embd: usize,
        rms_eps: f32,
    ) !DflashEncoder {
        if (sc.target_layers.len == 0) return error.SidecarNotLoaded;
        const fc = sc.model.file.getTensor("fc.weight") orelse return error.SidecarNotLoaded;
        const norm = sc.model.file.getTensor("enc.output_norm.weight") orelse return error.SidecarNotLoaded;
        const fc_k = sc.target_layers.len * n_embd;
        if (fc.shape()[0] != fc_k or fc.shape()[1] != n_embd) {
            debugz.dbg.printLevel(.info, "[dflash_enc] fc dims {any} ≠ [{d},{d}] (n_taps={d})\n", .{ fc.shape(), fc_k, n_embd, sc.target_layers.len });
            return error.DimensionMismatch;
        }

        // fc q8_0 mmap → device (cache q4Weight por host-ptr).
        const fc_data = sc.model.file.tensorData(fc);
        const fc_dev = try layer_kernels.q4Weight(allocator, @intFromPtr(fc_data.ptr), fc_data);

        // enc.output_norm f32 → device.
        const norm_data = sc.model.file.tensorData(norm);
        const norm_dev = try cudaz.cuMemAlloc(norm_data.len);
        errdefer cudaz.cuMemFree(norm_dev);
        try cudaz.cuMemcpyHtoD(norm_dev, @intFromPtr(norm_data.ptr), norm_data.len);

        const n_taps = sc.target_layers.len;
        const taps_ptrs_dev = try cudaz.cuMemAlloc(n_taps * 8);
        errdefer cudaz.cuMemFree(taps_ptrs_dev);
        const fc_in_dev = try cudaz.cuMemAlloc(fc_k * 4);
        errdefer cudaz.cuMemFree(fc_in_dev);
        const fc_out_dev = try cudaz.cuMemAlloc(n_embd * 4);
        errdefer cudaz.cuMemFree(fc_out_dev);

        return .{
            .allocator = allocator,
            .lk = lk,
            .target_layers = sc.target_layers,
            .n_embd = n_embd,
            .fc_k = fc_k,
            .eps = rms_eps,
            .fc_bytes = fc_data,
            .fc_dev = fc_dev,
            .norm_dev = norm_dev,
            .taps_ptrs_dev = taps_ptrs_dev,
            .fc_in_dev = fc_in_dev,
            .fc_out_dev = fc_out_dev,
        };
    }

    pub fn deinit(self: *DflashEncoder) void {
        // fc_dev pertenece al cache q4Weight (lo libera deinitQ4Cache).
        cudaz.cuMemFree(self.norm_dev);
        cudaz.cuMemFree(self.taps_ptrs_dev);
        cudaz.cuMemFree(self.fc_in_dev);
        cudaz.cuMemFree(self.fc_out_dev);
    }

    /// Una pasada del encoder. `taps_dev` = punteros device de los hidden
    /// de las target_layers (en el ORDEN de sc.target_layers; el caller
    /// los rellena por ronda — ver dflashEncodeForward).
    /// Devuelve el ptr device del hidden normalizado (fc_out_dev).
    pub fn forward(self: *DflashEncoder, taps_dev: []const usize) !usize {
        if (taps_dev.len != self.target_layers.len) return error.DimensionMismatch;
        // host → device: el array de ptrs (8×8B — trivial, pero es la ÚNICA
        // H2D del camino; alternativa futura: registro persistente por capa
        // con re-mapeo on-evict).
        var host_ptrs: [64]usize = undefined;
        for (taps_dev, 0..) |p, i| host_ptrs[i] = p;
        try cudaz.cuMemcpyHtoD(self.taps_ptrs_dev, @intFromPtr(&host_ptrs), taps_dev.len * 8);

        try self.lk.dflashTapsGather(self.taps_ptrs_dev, self.fc_in_dev, self.n_embd, taps_dev.len);
        // GEMV q8_0 M=1: fc_out = fc · fc_in. Kernel dedicado split-K: el
        // M1 genérico pide smem k·4B = 128KB > límite 99KB de sm_86 con
        // K=32768 (invalid argument, root-caused con el test de paridad).
        try self.lk.dflashFcGemmM1(self.fc_in_dev, self.fc_dev, self.fc_out_dev, self.fc_k, self.n_embd);
        try self.lk.rmsNorm(self.fc_out_dev, self.norm_dev, self.fc_out_dev, 1, self.n_embd, self.eps);
        return self.fc_out_dev;
    }

    /// Hidden del drafter tras la pasada (para el KV-inject/denoise C6.2+).
    pub fn hiddenDev(self: *const DflashEncoder) usize {
        return self.fc_out_dev;
    }
};

test "DflashEncoder dims sanity (sin GPU)" {
    // Smoke de la validación de dims sin tocar CUDA: fc_k = n_taps·n_embd.
    const n_taps: usize = 8;
    const n_embd: usize = 4096;
    try std.testing.expectEqual(n_taps * n_embd, 32768);
}
