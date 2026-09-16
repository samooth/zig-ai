//! Tipos de datos para KV-cache cuantizado
//! Define formatos, tensores cuantizados y metadatos de bloque

const std = @import("std");

/// Formatos de cuantización soportados (mapeo 1:1 con gguf.GgmlType cuantizados)
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
    /// Q4_1 (formato GGUF): bloque de 32, scale f16 + min f16 + 16 nibbles
    q4_1,
    /// Q5_0 (formato GGUF): bloque de 32, scale f16 + qh u32 + 16 nibbles
    q5_0,
    /// Q5_1 (formato GGUF): bloque de 32, scale f16 + min f16 + qh u32 + 16 nibbles
    q5_1,
    /// Q8_0 (formato GGUF): bloque de 32, scale f16 + 32 bytes int8
    q8_0,
    /// Q8_1 (formato GGUF): bloque de 32, scale f16 + min f16 + 32 bytes int8
    q8_1,
    // ── P0.3 lane-b2: tipos cache estándar adicionales (BeeLlama port) ──
    /// Q2_0S (formato BeeLlama Q2_0S): bloque de 32, scale f16 + qs[8].
    /// Variante SEGURA del Q2_0 upstream (block=64) — distinto nombre para
    /// no colisionar con el cuantizador de pesos (docs/BEELLAMA_PORT.md §P0.3).
    q2_0s,
    /// Q2_1 (formato GGUF): bloque de 32, scale f16 + min f16 + qs[8]
    q2_1,
    /// Q3_0 (formato GGUF): bloque de 32, scale f16 + qh[4] + qs[8]
    q3_0,
    /// Q3_1 (formato GGUF): bloque de 32, scale f16 + min f16 + qh[4] + qs[8]
    q3_1,
    /// Q6_0 (formato GGUF): bloque de 32, scale f16 + qh[8] + qs[16]
    q6_0,
    /// Q6_1 (formato GGUF): bloque de 32, scale f16 + min f16 + qh[8] + qs[16]
    q6_1,
    /// Q2_K (formato GGUF K-quants): super-bloque 256, 84 bytes
    q2_k,
    /// Q3_K (formato GGUF K-quants): super-bloque 256, 110 bytes
    q3_k,
    /// Q4_K (formato GGUF K-quants): super-bloque 256, 144 bytes
    q4_k,
    /// Q5_K (formato GGUF K-quants): super-bloque 256, 176 bytes
    q5_k,
    /// Q6_K (formato GGUF K-quants): super-bloque 256, 210 bytes
    q6_k,
    /// Q8_K (formato GGUF K-quants): super-bloque 256, 292 bytes
    q8_k,
    /// IQ1_S (formato GGUF I-quants): super-bloque 256, 50 bytes
    iq1_s,
    /// IQ1_M (formato GGUF I-quants): super-bloque 256, 56 bytes
    iq1_m,
    /// IQ2_XXS (formato GGUF I-quants): super-bloque 256, 66 bytes
    iq2_xxs,
    /// IQ2_XS (formato GGUF I-quants): super-bloque 256, 74 bytes
    iq2_xs,
    /// IQ2_S (formato GGUF I-quants): super-bloque 256, 82 bytes
    iq2_s,
    /// IQ3_XXS (formato GGUF I-quants): super-bloque 256, 98 bytes
    iq3_xxs,
    /// IQ3_S (formato GGUF I-quants): super-bloque 256, 110 bytes
    iq3_s,
    /// IQ4_XS (formato GGUF I-quants): super-bloque 256, 136 bytes
    iq4_xs,
    /// IQ4_NL (formato GGUF I-quants): super-bloque 32, 18 bytes
    iq4_nl,
    /// TQ1_0 (formato GGUF T-quants): super-bloque 256, 54 bytes
    tq1_0,
    /// TQ2_0 (formato GGUF T-quants): super-bloque 256, 66 bytes
    tq2_0,
    /// MXFP4 (formato GGUF): bloque de 32, 17 bytes (e8m0 + 16 nibbles)
    mxfp4,
    /// FP8 E4M3 (formato FreeToken): bloque de 128, escala fp32 por grupo de 128
    fp8,

    /// Bits por elemento lógico
    pub fn bitsPerElement(self: QuantFormat) u8 {
        return switch (self) {
            .fp16 => 16,
            .fp32 => 32,
            .int8_symmetric, .int8_asymmetric => 8,
            .int4 => 4,
            .q4_0, .q4_1, .q5_0, .q5_1 => 4,
            .q8_0, .q8_1 => 8,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 4, // efectivo
            .q2_0s, .q2_1 => 2,
            .q3_0, .q3_1 => 3,
            .q6_0, .q6_1 => 6,
            .iq1_s, .iq1_m => 1,
            .iq2_xxs, .iq2_xs, .iq2_s => 2,
            .iq3_xxs, .iq3_s => 3,
            .iq4_xs, .iq4_nl => 4,
            .tq1_0 => 1,
            .tq2_0 => 2,
            .mxfp4 => 4,
        };
    }

    /// Tamaño de bloque por defecto (elementos lógicos por bloque)
    pub fn defaultBlockSize(self: QuantFormat) usize {
        return switch (self) {
            .fp16, .fp32 => 1,
            .int8_symmetric, .int8_asymmetric => 64,
            .int4 => 64,
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
            .q2_0s, .q2_1, .q3_0, .q3_1, .q6_0, .q6_1 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            .iq1_s, .iq1_m, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq4_xs, .tq1_0, .tq2_0 => 256,
            .iq4_nl => 32,
            .mxfp4 => 32,
            .fp8 => 128,
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
            .int8_symmetric => 64 + 4, // 64 bytes + 1 scale f32
            .int8_asymmetric => 64 + 8, // 64 bytes + scale f32 + zp f32
            .int4 => 32 + 8, // 32 bytes (64 nibbles) + scale + zp
            .q4_0 => 18, // 2 bytes scale f16 + 16 bytes datos
            .q4_1 => 20, // 2 bytes scale f16 + 2 bytes min f16 + 16 nibbles
            .q5_0 => 22, // f16 d + u32 qh + 16 nibbles
            .q5_1 => 24, // f16 d + f16 m + u32 qh + 16 nibbles
            .q8_0 => 34, // 2 bytes scale f16 + 32 bytes datos
            .q8_1 => 36, // f16 d + f16 s + 32 i8
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 292,
            .q2_0s => 2 + 8, // scale f16 + qs[8] (2 bpw × 32 / 8)
            .q2_1 => 2 + 2 + 8, // scale + min + qs[8]
            .q3_0 => 2 + 4 + 8, // scale + qh[4] + qs[8] (2 bpw × 32 / 8)
            .q3_1 => 2 + 2 + 4 + 8, // scale + min + qh[4] + qs[8]
            .q6_0 => 2 + 8 + 16, // scale + qh[8] (2b/elem) + qs[16] (4b/elem)
            .q6_1 => 2 + 2 + 8 + 16, // scale + min + qh[8] + qs[16]
            .iq1_s => 50,
            .iq1_m => 56,
            .iq2_xxs => 66,
            .iq2_xs => 74,
            .iq2_s => 82,
            .iq3_xxs => 98,
            .iq3_s => 110,
            .iq4_xs => 136,
            .iq4_nl => 18,
            .tq1_0 => 54,
            .tq2_0 => 66,
            .mxfp4 => 17,
            .fp8 => 128 + 4, // 128 FP8 elements + 1 fp32 scale per 128 elements
        };
    }

    /// Parsea un nombre de formato (como en llama.cpp: q8_0, q4_0, fp16, ...).
    /// Devuelve null si el nombre no es reconocido.
    pub fn fromString(s: []const u8) ?QuantFormat {
        if (std.mem.eql(u8, s, "q8_0")) return .q8_0;
        if (std.mem.eql(u8, s, "q4_0")) return .q4_0;
        if (std.mem.eql(u8, s, "q4_1")) return .q4_1;
        if (std.mem.eql(u8, s, "q5_0")) return .q5_0;
        if (std.mem.eql(u8, s, "q5_1")) return .q5_1;
        if (std.mem.eql(u8, s, "q8_1")) return .q8_1;
        if (std.mem.eql(u8, s, "q4_k")) return .q4_k;
        if (std.mem.eql(u8, s, "q5_k")) return .q5_k;
        if (std.mem.eql(u8, s, "q6_k")) return .q6_k;
        if (std.mem.eql(u8, s, "q8_k")) return .q8_k;
        if (std.mem.eql(u8, s, "q3_k")) return .q3_k;
        if (std.mem.eql(u8, s, "q2_k")) return .q2_k;
        // Lane-b2 P0.3: tipos cache estándar adicionales (BeeLlama port).
        // q2_0s es el alias "seguro" de Bee (block=32) — distinto del q2_0
        // upstream (block=64) que NO soportamos para evitar colisión con el
        // cuantizador de pesos (ver docs/BEELLAMA_PORT.md §P0.3).
        if (std.mem.eql(u8, s, "q2_0s")) return .q2_0s;
        if (std.mem.eql(u8, s, "q2_0")) return .q2_0s; // alias Bee para q2_0s
        if (std.mem.eql(u8, s, "q2_1")) return .q2_1;
        if (std.mem.eql(u8, s, "q3_0")) return .q3_0;
        if (std.mem.eql(u8, s, "q3_1")) return .q3_1;
        if (std.mem.eql(u8, s, "q6_0")) return .q6_0;
        if (std.mem.eql(u8, s, "q6_1")) return .q6_1;
        if (std.mem.eql(u8, s, "iq4_xs")) return .iq4_xs;
        if (std.mem.eql(u8, s, "iq3_s")) return .iq3_s;
        if (std.mem.eql(u8, s, "iq4_nl")) return .iq4_nl;
        if (std.mem.eql(u8, s, "iq2_xxs")) return .iq2_xxs;
        if (std.mem.eql(u8, s, "iq2_xs")) return .iq2_xs;
        if (std.mem.eql(u8, s, "iq3_xxs")) return .iq3_xxs;
        if (std.mem.eql(u8, s, "iq1_s")) return .iq1_s;
        if (std.mem.eql(u8, s, "iq2_s")) return .iq2_s;
        if (std.mem.eql(u8, s, "iq1_m")) return .iq1_m;
        if (std.mem.eql(u8, s, "tq1_0")) return .tq1_0;
        if (std.mem.eql(u8, s, "tq2_0")) return .tq2_0;
        if (std.mem.eql(u8, s, "mxfp4")) return .mxfp4;
        if (std.mem.eql(u8, s, "fp8")) return .fp8;
        if (std.mem.eql(u8, s, "int4")) return .int4;
        if (std.mem.eql(u8, s, "int8") or std.mem.eql(u8, s, "int8_sym")) return .int8_symmetric;
        if (std.mem.eql(u8, s, "int8_asym")) return .int8_asymmetric;
        if (std.mem.eql(u8, s, "fp16")) return .fp16;
        if (std.mem.eql(u8, s, "fp32") or std.mem.eql(u8, s, "f32")) return .fp32;
        if (std.mem.eql(u8, s, "f16")) return .fp16;
        if (std.mem.eql(u8, s, "none")) return .fp16;
        return null;
    }

    /// Nombre legible del formato (para logging).
    pub fn toString(self: QuantFormat) []const u8 {
        return switch (self) {
            .fp16 => "f16",
            .fp32 => "f32",
            .int8_symmetric => "int8_sym",
            .int8_asymmetric => "int8_asym",
            .int4 => "int4",
            .q4_0 => "q4_0",
            .q4_1 => "q4_1",
            .q5_0 => "q5_0",
            .q5_1 => "q5_1",
            .q8_0 => "q8_0",
            .q8_1 => "q8_1",
            .q2_k => "q2_k",
            .q3_k => "q3_k",
            .q4_k => "q4_k",
            .q5_k => "q5_k",
            .q6_k => "q6_k",
            .q8_k => "q8_k",
            // Lane-b2 P0.3: cache types adicionales.
            .q2_0s => "q2_0s",
            .q2_1 => "q2_1",
            .q3_0 => "q3_0",
            .q3_1 => "q3_1",
            .q6_0 => "q6_0",
            .q6_1 => "q6_1",
            .iq1_s => "iq1_s",
            .iq1_m => "iq1_m",
            .iq2_xxs => "iq2_xxs",
            .iq2_xs => "iq2_xs",
            .iq2_s => "iq2_s",
            .iq3_xxs => "iq3_xxs",
            .iq3_s => "iq3_s",
            .iq4_xs => "iq4_xs",
            .iq4_nl => "iq4_nl",
            .tq1_0 => "tq1_0",
            .tq2_0 => "tq2_0",
            .mxfp4 => "mxfp4",
            .fp8 => "fp8",
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
    /// 9.1 (lane-c C-1): bits del payload K del record KVarN. >0 activa el
    /// store KVarN para K (k_format se ignora en ese camino — KVarN es un
    /// formato de TILE de 128 tokens, no de bloques por token). Valores
    /// válidos: 2|3|4|5|6|8 (kvarn.valid_bits).
    kvarn_k_bits: u8 = 0,
    /// 9.1 (lane-c C-1): bits del payload V del record KVarN. >0 activa el
    /// store KVarN para V. Puede diferir de k_bits (p.ej. kvarn4v6).
    kvarn_v_bits: u8 = 0,
};

/// ── Lane-b2 P0.2 (KVCPT): cola exacta F16/BF16 ──
/// Tipo del slot exacto. Duplicado de tail_request.ExactType para evitar
/// import circular; el namespace `tail_request` re-exporta este tipo.
pub const ExactType = enum(u8) {
    /// Sin tipo explícito → usar F16 si KVarN, BF16 si Q*.
    default,
    f16,
    bf16,
};

/// Configuración global de KV-cache
pub const KVCacheConfig = struct {
    /// Número de capas
    num_layers: u32,
    /// Número de cabezas de atención
    num_heads: u32,
    /// Número de cabezas K/V (GQA; <= num_heads)
    num_kv_heads: u32,
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
    /// Lane-b2 P0.2: tokens exactos en ring (0 = desactivar). KVarN
    /// redondea a múltiplos de 128 (group size).
    tail_tokens: u32 = 0,
    /// Lane-b2 P0.2: tipo del slot exacto. `default` resuelve a f16 si la
    /// config usa KVarN, bf16 en caso contrario.
    tail_type: ExactType = .default,

    /// Configuración por defecto (mixta: K=Q4_K, V=Q8_0 para mejor calidad/compresión)
    pub fn default(num_layers: u32, num_heads: u32, head_dim: u32, max_seq_len: u32) KVCacheConfig {
        return .{
            .num_layers = num_layers,
            .num_heads = num_heads,
            .num_kv_heads = num_heads,
            .head_dim = head_dim,
            .max_seq_len = max_seq_len,
            .layer_configs = null,
            .use_gpu_dequant = true,
            .enable_prefetch = true,
            .enable_streaming = true,
        };
    }

    /// Configuración agresiva (máxima compresión: K=Q4_K, V=Q4_K)
    pub fn aggressive(num_layers: u32, num_heads: u32, head_dim: u32, max_seq_len: u32) KVCacheConfig {
        return .{
            .num_layers = num_layers,
            .num_heads = num_heads,
            .num_kv_heads = num_heads,
            .head_dim = head_dim,
            .max_seq_len = max_seq_len,
            .layer_configs = null,
            .use_gpu_dequant = true,
            .enable_prefetch = true,
            .enable_streaming = true,
        };
    }

    /// Configuración alta calidad (K=Q6_K, V=Q8_0)
    pub fn highQuality(num_layers: u32, num_heads: u32, head_dim: u32, max_seq_len: u32) KVCacheConfig {
        return .{
            .num_layers = num_layers,
            .num_heads = num_heads,
            .num_kv_heads = num_heads,
            .head_dim = head_dim,
            .max_seq_len = max_seq_len,
            .layer_configs = null,
            .use_gpu_dequant = true,
            .enable_prefetch = true,
            .enable_streaming = true,
        };
    }

    /// Configuración I-quants (mejor calidad/compresión: K=IQ4_XS, V=IQ4_XS)
    pub fn iQuants(num_layers: u32, num_heads: u32, head_dim: u32, max_seq_len: u32) KVCacheConfig {
        return .{
            .num_layers = num_layers,
            .num_heads = num_heads,
            .num_kv_heads = num_heads,
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

    /// Mapea un head de query a su head K/V (GQA)
    pub fn qHeadToKvHead(self: KVCacheConfig, q_head: usize) usize {
        const group_size = self.num_heads / self.num_kv_heads;
        return q_head / group_size;
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

/// Configuración de offload CPU
pub const OffloadConfig = struct {
    /// Habilitar offload a CPU
    enabled: bool = true,
    /// VRAM libre mínima antes de hacer offload (MB)
    min_free_vram_mb: usize = 512,
    /// Edad mínima en tokens antes de considerar offload
    min_token_age: u32 = 64,
    /// Número máximo de bloques en CPU
    max_cpu_blocks: usize = 1024,
    /// Stream dedicado para offload/reload
    use_dedicated_stream: bool = true,
};

test {
    std.testing.refAllDecls(@This());
}
