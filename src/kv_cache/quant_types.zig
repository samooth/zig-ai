//! Tipos de datos para KV-cache cuantizado
//! Define formatos, tensores cuantizados y metadatos de bloque

const std = @import("std");

/// Formatos de cuantización soportados
pub const QuantFormat = enum {
    /// FP16 sin cuantizar (baseline)
    fp16,
    /// FP32 sin cuantizar (precisión completa)
    fp32,
    /// INT8 simétrico por bloque: scale por bloque, cero en 0
    int8_symmetric,
    /// INT8 asimétrico por bloque: scale + zero_point por bloque
    int8_asymmetric,
    /// INT4 asimétrico por bloque: 2 valores por byte
    int4,
    /// Q4_0 (formato GGUF): bloque de 32, scale f16 + 16 bytes de nibbles
    q4_0,
    /// Q8_0 (formato GGUF): bloque de 32, scale f16 + 32 bytes
    q8_0,

    /// Bits por elemento lógico
    pub fn bitsPerElement(self: QuantFormat) u8 {
        return switch (self) {
            .fp16 => 16,
            .fp32 => 32,
            .int8_symmetric, .int8_asymmetric => 8,
            .int4, .q4_0 => 4,
            .q8_0 => 8,
        };
    }

    /// Tamaño de bloque por defecto
    pub fn defaultBlockSize(self: QuantFormat) usize {
        return switch (self) {
            .fp16, .fp32 => 1,
            .int8_symmetric, .int8_asymmetric => 64,
            .int4 => 64,
            .q4_0, .q8_0 => 32,
        };
    }

    /// Requiere metadatos de escala
    pub fn hasScales(self: QuantFormat) bool {
        return switch (self) {
            .fp16, .fp32 => false,
            else => true,
        };
    }

    /// Requiere zero_points
    pub fn hasZeroPoints(self: QuantFormat) bool {
        return switch (self) {
            .int8_asymmetric, .int4 => true,
            else => false,
        };
    }

    /// Bytes por bloque (incluyendo metadatos)
    pub fn bytesPerBlock(self: QuantFormat) usize {
        return switch (self) {
            .fp16 => 2,
            .fp32 => 4,
            .int8_symmetric => 64 + 4,   // 64 bytes + 1 scale f32
            .int8_asymmetric => 64 + 8,  // 64 bytes + scale f32 + zp f32
            .int4 => 32 + 8,             // 32 bytes (64 nibbles) + scale + zp
            .q4_0 => 18,                 // 2 bytes scale f16 + 16 bytes datos
            .q8_0 => 34,                 // 2 bytes scale f16 + 32 bytes datos
        };
    }
};

/// Tensor cuantizado en host o device
pub const QuantizedTensor = struct {
    format: QuantFormat,
    /// Datos cuantizados crudos
    raw: []const u8,
    /// Metadatos de escala (por bloque), null si no aplica
    scales: ?[]const f32,
    /// Zero points (por bloque), null si no aplica
    zero_points: ?[]const f32,
    /// Número de elementos lógicos
    num_elements: usize,
    /// Tamaño de bloque usado
    block_size: usize,
    /// Número de bloques
    num_blocks: usize,

    pub fn init(
        format: QuantFormat,
        raw: []const u8,
        scales: ?[]const f32,
        zero_points: ?[]const f32,
        num_elements: usize,
        block_size: usize,
    ) QuantizedTensor {
        const num_blocks = (num_elements + block_size - 1) / block_size;
        return .{
            .format = format,
            .raw = raw,
            .scales = scales,
            .zero_points = zero_points,
            .num_elements = num_elements,
            .block_size = block_size,
            .num_blocks = num_blocks,
        };
    }

    /// Bytes totales ocupados incluyendo metadatos
    pub fn totalBytes(self: QuantizedTensor) usize {
        var total = self.raw.len;
        if (self.scales) |s| total += s.len * @sizeOf(f32);
        if (self.zero_points) |z| total += z.len * @sizeOf(f32);
        return total;
    }

    /// Ratio de compresión vs FP16
    pub fn compressionRatio(self: QuantizedTensor) f32 {
        const fp16_bytes = self.num_elements * 2;
        return @as(f32, @floatFromInt(fp16_bytes)) / @as(f32, @floatFromInt(self.totalBytes()));
    }
};

/// Descriptor de un bloque de KV-cache
pub const KVBlockDescriptor = struct {
    /// Índice de capa (layer)
    layer_idx: u32,
    /// Índice de cabeza de atención
    head_idx: u32,
    /// Posición de inicio en la secuencia
    seq_start: u32,
    /// Longitud de la secuencia en este bloque
    seq_len: u32,
    /// Dimensión de embedding por cabeza
    head_dim: u32,
    /// Formato de cuantización
    format: QuantFormat,
    /// Offset en el buffer contiguo
    byte_offset: usize,
    /// Tamaño en bytes
    byte_size: usize,
};

/// Configuración de cuantización por capa
pub const LayerQuantConfig = struct {
    /// Formato para Key cache
    k_format: QuantFormat,
    /// Formato para Value cache
    v_format: QuantFormat,
    /// Tamaño de bloque para K
    k_block_size: usize,
    /// Tamaño de bloque para V
    v_block_size: usize,
    /// Umbral de activación de cuantización (siempre cuantizar si null)
    quant_threshold: ?usize,
};

/// Configuración global de KV-cache
pub const KVCacheConfig = struct {
    /// Número de capas
    num_layers: u32,
    /// Número de cabezas de atención
    num_heads: u32,
    /// Dimensión por cabeza
    head_dim: u32,
    /// Longitud máxima de secuencia
    max_seq_len: u32,
    /// Configuración de cuantización por capa (null = usar default)
    layer_configs: ?[]const LayerQuantConfig,
    /// Usar GPU para de-cuantización
    use_gpu_dequant: bool,
    /// Prefetch de bloques anticipado
    enable_prefetch: bool,
    /// Overlap compute/memcpy con streams
    enable_streaming: bool,

    /// Configuración por defecto (mixta: K=Q4_0, V=Q8_0)
    pub fn default(num_layers: u32, num_heads: u32, head_dim: u32, max_seq_len: u32) KVCacheConfig {
        return .{
            .num_layers = num_layers,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .max_seq_len = max_seq_len,
            .layer_configs = null,
            .use_gpu_dequant = true,
            .enable_prefetch = true,
            .enable_streaming = true,
        };
    }

    /// Configuración agresiva (máxima compresión)
    pub fn aggressive(num_layers: u32, num_heads: u32, head_dim: u32, max_seq_len: u32) KVCacheConfig {
        return .{
            .num_layers = num_layers,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .max_seq_len = max_seq_len,
            .layer_configs = null,
            .use_gpu_dequant = true,
            .enable_prefetch = true,
            .enable_streaming = true,
        };
    }

    /// Tamaño total estimado en bytes para una configuración dada
    pub fn estimatedSize(self: KVCacheConfig, format: QuantFormat) usize {
        const elements_per_layer = @as(usize, self.num_heads) * @as(usize, self.max_seq_len) * @as(usize, self.head_dim);
        const elements_total = elements_per_layer * self.num_layers * 2; // K + V
        const bits = format.bitsPerElement();
        return (elements_total * bits) / 8;
    }
};

/// Estado de un slot de cache
pub const CacheSlot = struct {
    /// Índice del slot
    idx: u32,
    /// Ocupado
    occupied: bool,
    /// Número de referencias activas
    ref_count: u32,
    /// Timestamp de último acceso (para LRU)
    last_access: u64,
    /// Descriptor del bloque
    descriptor: KVBlockDescriptor,
};
