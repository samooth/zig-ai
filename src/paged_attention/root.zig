const std = @import("std");
const kvcache = @import("kv_cache");
pub const QuantFormat = kvcache.QuantFormat;
pub const kv_quant = kvcache.kv_quant;

pub const Block = @import("block.zig").Block;
pub const hashTokens = @import("block.zig").hashTokens;
pub const BlockAllocator = @import("allocator.zig").BlockAllocator;
pub const CpuOffloadManager = @import("allocator.zig").CpuOffloadManager;
pub const BlockTable = @import("block_table.zig").BlockTable;
pub const PagedKVCache = @import("paged_kv_cache.zig").PagedKVCache;
pub const PagedAttention = @import("attention.zig").PagedAttention;
pub const Scheduler = @import("scheduler.zig").Scheduler;
pub const PrefixCache = @import("prefix_cache.zig").PrefixCache;
pub const Sequence = @import("scheduler.zig").Sequence;
pub const Request = @import("scheduler.zig").Request;
pub const PagedAttentionGpu = @import("gpu_kernels.zig").PagedAttentionGpu;
/// G1 (lane-b1): setter de test para el opt-in ZIG_AI_FASPLIT (cache de
/// proceso — ver test_paged_attention_gpu "G1 split-K decode").
pub const fasplitForceForTest = @import("gpu_kernels.zig").fasplitForceForTest;
pub const GpuBlockPool = @import("gpu_kernels.zig").GpuBlockPool;
pub const PagedGpuBlockPool = @import("paged_gpu_pool.zig").PagedGpuBlockPool;

pub const DType = enum { f32, f16, bf16 };

pub const PagedConfig = struct {
    block_size: usize = 16,
    num_blocks: usize = 1024,
    head_dim: usize = 128,
    num_kv_heads: usize = 8,
    num_q_heads: usize = 32,
    dtype: DType = .f16,
    /// KV cache quantization format K (llama.cpp --cache-type-k; .fp16 = off)
    quant_k: QuantFormat = .fp16,
    /// KV cache quantization format V (llama.cpp --cache-type-v; .fp16 = off)
    quant_v: QuantFormat = .fp16,
    enable_prefix_cache: bool = true,
    enable_cpu_offload: bool = false,
    enable_proactive_evict: bool = false,
    proactive_evict_min_free: usize = 4,
    proactive_evict_stale_age: u64 = 32,
    max_seq_len: usize = 8192,
    max_batch_size: usize = 64,

    /// Total elements per block for K or V (block_size * num_kv_heads * head_dim)
    pub fn blockElems(self: PagedConfig) usize {
        return self.block_size * self.num_kv_heads * self.head_dim;
    }

    /// Number of 32-element quant blocks per K or V block
    pub fn quantBlocksPerBlock(self: PagedConfig) usize {
        return (self.blockElems() + 31) / 32;
    }

    /// Bytes for quantized K or V data per block (including scale metadata)
    pub fn quantDataBytes(self: PagedConfig, format: QuantFormat) usize {
        return kv_quant.quantBytes(format, self.blockElems());
    }

    /// Total bytes per block for K + V (quantized data)
    pub fn quantBlockBytes(self: PagedConfig) usize {
        return self.quantDataBytes(self.quant_k) + self.quantDataBytes(self.quant_v);
    }

    /// Bytes for scales per block (K or V)
    /// q8_0: 1 f16 scale per 32 elements = quantBlocksPerBlock() * 2 bytes
    pub fn scaleBytes(self: PagedConfig, format: QuantFormat) usize {
        if (!format.hasScales()) return 0;
        return self.quantBlocksPerBlock() * @sizeOf(f16);
    }

    /// Total bytes per block for K+V tal como los indexan los kernels fusionados:
    /// [K: quantDataBytes(k)][V: quantDataBytes(v)] con escalas EMBEBIDAS
    /// (Contrato C1, PLAN_MAESTRO). SIN cola de escalas separada: el stride del
    /// pool debe coincidir exactamente con el `phys * (k_bytes + v_bytes)` que
    /// usan decode/prefill/append; cualquier cola extra desalinea bloques
    /// físicos ≥1 (los kernels leen datos del vecino anterior).
    /// Las escalas legacy de `k_scales/v_scales` viven en arrays propios del
    /// BlockAllocator (totalScaleBytes), no en el pool.
    pub fn totalBlockBytes(self: PagedConfig) usize {
        return self.quantBlockBytes();
    }

    /// Total scale bytes for K+V
    pub fn totalScaleBytes(self: PagedConfig) usize {
        return self.scaleBytes(self.quant_k) + self.scaleBytes(self.quant_v);
    }
};

pub const Stats = struct {
    blocks_allocated: usize = 0,
    blocks_free: usize = 0,
    blocks_shared: usize = 0,
    sequences_active: usize = 0,
    prefix_hits: usize = 0,
    prefix_misses: usize = 0,
    prefix_evictions: usize = 0,
    prefix_proactive_evictions: usize = 0,
    prefix_hit_rate: f64 = 0.0,
};
