//! zig-kv-cache-gpu
//! KV-cache con cuantización y de-cuantización GPU para inferencia LLM
//!
//! Uso básico:
//! ```zig
//! const kvc = @import("kv_cache");
//!
//! var manager = try kvc.KVCacheManager.init(allocator, config, 1024);
//! defer manager.deinit();
//!
//! try manager.createSequence(1);
//! try manager.appendTokens(1, 0, 0, k_quantized, v_quantized);
//! ```

const std = @import("std");

pub const quant_types = @import("kv_cache/quant_types.zig");
pub const kv_quant = @import("kv_cache/kv_quant.zig");
pub const allocator = @import("kv_cache/allocator.zig");
pub const gpu_dequant = @import("kv_cache/gpu_dequant.zig");
pub const kv_cache_manager = @import("kv_cache/kv_cache_manager.zig");
pub const stream = @import("kv_cache/stream.zig");
pub const flash_attention = @import("kv_cache/flash_attention.zig");
pub const offload = @import("kv_cache/offload.zig");
pub const iq_grids = @import("kv_cache/iq_grids.zig");
/// §11.1: encoder iq1_m canónico (port 1:1 llama.cpp quantize_row_iq1_m_impl).
pub const iq1m_encoder = @import("kv_cache/iq1m_encoder.zig");
/// Lane-B2: KVarN record layout + CPU reference encode/decode (C1 contract).
pub const kvarn = @import("kv_cache/kvarn.zig");
/// Lane-B2: KVCPT (KV-cache precision tail) — política de cola exacta F16/BF16.
pub const tail_request = @import("kv_cache/tail_request.zig");
/// Lane-B2 P2.3: Checkpoint transaccional de KV cache (formato binario v3).
pub const checkpoint = @import("kv_cache/checkpoint.zig");
/// Lane-B2 P2.3: Restore transaccional con rollback atómico.
pub const restore = @import("kv_cache/restore.zig");
/// Lane-KVC (paso 1 KV-Codec): captura de trazas K/V del forward CPU.
pub const kv_trace = @import("kv_cache/kv_trace.zig");
/// KT-B (lane-f): runtime opt-in KV-transfer source→target (mappers .ktb).
pub const kt_transfer = @import("kv_cache/kt_transfer.zig");

// Re-exportar tipos principales
pub const QuantFormat = quant_types.QuantFormat;
pub const QuantizedTensor = quant_types.QuantizedTensor;
pub const KVCacheConfig = quant_types.KVCacheConfig;
pub const LayerQuantConfig = quant_types.LayerQuantConfig;
pub const KVBlockDescriptor = quant_types.KVBlockDescriptor;
pub const CacheSlot = quant_types.CacheSlot;
pub const OffloadConfig = quant_types.OffloadConfig;

pub const KVPoolAllocator = allocator.KVPoolAllocator;
pub const AllocStrategy = allocator.AllocStrategy;

pub const GpuDequantBuffers = gpu_dequant.GpuDequantBuffers;
pub const GpuDequantEngine = gpu_dequant.GpuDequantEngine;

pub const KVCacheManager = kv_cache_manager.KVCacheManager;
pub const SequenceState = kv_cache_manager.SequenceState;

pub const StreamRing = stream.StreamRing;
pub const PrefetchPipeline = stream.PrefetchPipeline;
pub const StreamTimer = stream.StreamTimer;

pub const FlashAttentionOp = flash_attention.FlashAttentionOp;
pub const GPUTensor = flash_attention.GPUTensor;
pub const KVFlashAttention = flash_attention.KVFlashAttention;
pub const QuantizedGPUTensor = flash_attention.QuantizedGPUTensor;
pub const ShapeUtils = flash_attention.ShapeUtils;

pub const CpuOffloadPipeline = offload.CpuOffloadPipeline;
pub const OffloadEntry = offload.OffloadEntry;
pub const OffloadBlockState = offload.OffloadBlockState;

/// Versión del paquete
pub const version = "0.1.0";

test {
    std.testing.refAllDecls(@This());
}
