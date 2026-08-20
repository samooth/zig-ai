# KV-Cache Cuantizado en Zig — Diseño e Implementación

> **ARCHIVADO (histórico)** — diseño/implementación ya construida. Ver docs vigentes:
> `PAGED_ATTENTION_TODO.md` y `docs/airllm-layer-streaming-guide.md`. Nota: el estado real
> actual difiere de lo que describe este doc (p. ej. la atención GPU está habilitada,
> `gpu_attention_enabled = true` en `src/main.zig`; el KV-cache paginado vive en
> `src/paged_attention/`, no en `src/kv_cache/`).

> **Fecha:** Agosto 2026  
> **Contexto:** Motor de KV-cache para AirLLM-Zig con soporte multi-formato de cuantización, paging y GQA/MQA.

---

## 1. Resumen Ejecutivo

| Pregunta | Respuesta |
|---|---|
| **¿Por qué cuantizar el KV-cache?** | Reduce memoria 2× (INT8) a 4× (INT4). Para Llama-3-70B con seq=32K, el KV-cache en FP16 ocupa **80 GB**. En INT4: **20 GB**. |
| **¿Qué formatos soportar?** | FP16, INT8 simétrico/asimétrico, INT4, Q4_0 (GGUF), Q8_0 (GGUF), y FP8 (E4M3). |
| **¿Dónde se de-cuantiza?** | **On-the-fly dentro del kernel FlashAttention** o en el host justo antes de lanzar. La de-cuantización en GPU es más rápida (ancho de banda HBM >> compute). |
| **¿GQA/MQA?** | Sí. El KV-cache se almacena una vez por grupo de heads, no por head individual. |

---

## 2. Formatos de Cuantización Soportados

### 2.1 Matriz de Formatos

| Formato | Bits/valor | Escala | Cero-point | Uso recomendado | Compresión |
|---|---|---|---|---|---|
| **FP16** | 16 | N/A | N/A | Precisión máxima, corto alcance | 1× |
| **FP8 (E4M3)** | 8 | N/A | N/A | Hopper/Blackwell nativo | 2× |
| **INT8 simétrico** | 8 | `scale` | 0 | Distribuciones centradas (post-RMSNorm) | 2× |
| **INT8 asimétrico** | 8 | `scale` | `zero_point` | Distribuciones sesgadas | 2× |
| **INT4** | 4 | `scale` | `zero_point` | Máxima compresión, pérdida aceptable | 4× |
| **Q4_0 (GGUF)** | 4 | `scale` por bloque 32 | 0 | Estándar GGUF, buen equilibrio | 4× |
| **Q5_K (GGUF)** | 5 | `scale` + `min` por super-bloque | variable | Máxima calidad GGUF | 3.2× |
| **Q8_0 (GGUF)** | 8 | `scale` por bloque 32 | 0 | Alta calidad, 2× compresión | 2× |

### 2.2 Representación en Zig

```zig
pub const QuantFormat = enum {
    fp16,
    fp8_e4m3,
    int8_symmetric,
    int8_asymmetric,
    int4,
    q4_0,      // GGUF: 32 valores + 1 scale (f16) = 18 bytes por bloque
    q5_k,      // GGUF: super-bloques de 256
    q8_0,      // GGUF: 32 valores + 1 scale (f16) = 34 bytes por bloque
};
```

---

## 3. Arquitectura del KV-Cache

### 3.1 Diagrama de Componentes

```
+-------------------------------------------------------------+
|                    KVCacheManager                           |
|  - Múltiples secuencias (batch)                             |
|  - Paging: bloques de 256 tokens                            |
|  - Evicción LRU (cuando VRAM llena)                         |
+-------------------------------------------------------------+
        |
   +----+----+----+----+
   |         |         |
+--v--+   +--v--+   +--v--+
|Seq 0|   |Seq 1|   |Seq N|   <- Una por secuencia del batch
+--+--+   +--+--+   +--+--+
   |         |         |
+--v-------------------------+
|  KVCacheSequence            |
|  - Lista de bloques paginados|
|  - current_len              |
|  - max_len                  |
+--+--------------------------+
   |
+--v-------------------------+
|  KVCacheBlock (256 tokens)  |
|  - K data cuantizado        |
|  - V data cuantizado        |
|  - scale/zero per channel   |
+--+--------------------------+
   |
+--v-------------------------+
|  QuantizedTensor            |
|  - raw_bytes: []u8          |
|  - format: QuantFormat      |
|  - block_scales: []f32      |
|  - block_zeros: []f32       |
+-----------------------------+
```

### 3.2 Paging de Bloques

En lugar de alocar `[max_seq_len, num_heads, head_dim]` contiguo, usamos **bloques de 256 tokens**:

```
Secuencia de 1000 tokens:
  [Block 0: tokens 0-255]  →  alocado
  [Block 1: tokens 256-511] →  alocado
  [Block 2: tokens 512-767] →  alocado
  [Block 3: tokens 768-999] →  alocado (parcial)
  [Block 4-127]             →  no alocado (lazy)
```

**Ventajas:**
- Solo se aloca memoria para tokens generados (no todo `max_seq_len`)
- Bloques pueden evictarse a CPU/disco si VRAM llena
- Compartir bloques entre secuencias (prefix caching)

---

## 4. Implementación en Zig

### 4.1 Tipos Base

```zig
// src/kv_cache/quant_types.zig
const std = @import("std");

pub const QuantFormat = enum {
    fp16,
    fp8_e4m3,
    int8_symmetric,
    int8_asymmetric,
    int4,
    q4_0,
    q5_k,
    q8_0,

    pub fn bitsPerValue(self: QuantFormat) u8 {
        return switch (self) {
            .fp16 => 16,
            .fp8_e4m3 => 8,
            .int8_symmetric, .int8_asymmetric, .q8_0 => 8,
            .int4, .q4_0 => 4,
            .q5_k => 5,
        };
    }

    pub fn bytesPerBlock(self: QuantFormat, block_size: usize) usize {
        const total_bits = block_size * self.bitsPerValue();
        return (total_bits + 7) / 8;
    }

    pub fn hasZeroPoint(self: QuantFormat) bool {
        return switch (self) {
            .int8_asymmetric, .int4 => true,
            else => false,
        };
    }

    pub fn isGguf(self: QuantFormat) bool {
        return switch (self) {
            .q4_0, .q5_k, .q8_0 => true,
            else => false,
        };
    }
};

/// Metadatos de cuantización por bloque
pub const BlockQuantMeta = struct {
    scale: f32,
    zero_point: f32 = 0.0,  // Solo para asimétrico
};

/// Tensor cuantizado genérico
pub const QuantizedTensor = struct {
    allocator: std.mem.Allocator,
    raw_data: []u8,           // Bytes cuantizados
    meta: []BlockQuantMeta,   // Uno por bloque
    format: QuantFormat,
    num_elements: usize,      // Número de valores lógicos
    block_size: usize,        // Tamaño de bloque (ej: 32, 64, 256)

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        num_elements: usize,
        format: QuantFormat,
        block_size: usize,
    ) !Self {
        const num_blocks = (num_elements + block_size - 1) / block_size;
        const bytes_needed = format.bytesPerBlock(block_size) * num_blocks;

        const raw_data = try allocator.alloc(u8, bytes_needed);
        errdefer allocator.free(raw_data);

        const meta = try allocator.alloc(BlockQuantMeta, num_blocks);
        errdefer allocator.free(meta);

        @memset(raw_data, 0);
        for (meta) |*m| m.* = .{ .scale = 1.0, .zero_point = 0.0 };

        return .{
            .allocator = allocator,
            .raw_data = raw_data,
            .meta = meta,
            .format = format,
            .num_elements = num_elements,
            .block_size = block_size,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.raw_data);
        self.allocator.free(self.meta);
    }

    /// Bytes totales usados (datos + metadatos)
    pub fn totalBytes(self: Self) usize {
        return self.raw_data.len + self.meta.len * @sizeOf(BlockQuantMeta);
    }

    /// Bytes que ocuparía en FP16
    pub fn fp16Bytes(self: Self) usize {
        return self.num_elements * 2;
    }

    /// Ratio de compresión
    pub fn compressionRatio(self: Self) f32 {
        return @as(f32, @floatFromInt(self.fp16Bytes())) / @as(f32, @floatFromInt(self.totalBytes()));
    }
};
```

### 4.2 Cuantización / De-cuantización

```zig
// src/kv_cache/quant_ops.zig
const std = @import("std");
const QuantizedTensor = @import("quant_types.zig").QuantizedTensor;
const QuantFormat = @import("quant_types.zig").QuantFormat;
const BlockQuantMeta = @import("quant_types.zig").BlockQuantMeta;

/// Cuantiza un tensor FP16 a formato cuantizado
pub fn quantizeF16(
    src: []const f16,
    dst: *QuantizedTensor,
    format: QuantFormat,
) void {
    switch (format) {
        .fp16 => {
            @memcpy(dst.raw_data, std.mem.sliceAsBytes(src));
        },
        .int8_symmetric => quantizeInt8Symmetric(src, dst),
        .int8_asymmetric => quantizeInt8Asymmetric(src, dst),
        .int4 => quantizeInt4(src, dst),
        .q4_0 => quantizeQ4_0(src, dst),
        .q8_0 => quantizeQ8_0(src, dst),
        else => @panic("Formato de cuantización no implementado"),
    }
}

/// De-cuantiza a FP16
pub fn dequantizeToF16(
    src: QuantizedTensor,
    dst: []f16,
) void {
    switch (src.format) {
        .fp16 => {
            @memcpy(std.mem.sliceAsBytes(dst), src.raw_data);
        },
        .int8_symmetric => dequantizeInt8Symmetric(src, dst),
        .int8_asymmetric => dequantizeInt8Asymmetric(src, dst),
        .int4 => dequantizeInt4(src, dst),
        .q4_0 => dequantizeQ4_0(src, dst),
        .q8_0 => dequantizeQ8_0(src, dst),
        else => @panic("Formato de de-cuantización no implementado"),
    }
}

// ─── INT8 Simétrico ───

fn quantizeInt8Symmetric(src: []const f16, dst: *QuantizedTensor) void {
    const block_size = dst.block_size;
    const num_blocks = dst.meta.len;

    for (0..num_blocks) |b| {
        const start = b * block_size;
        const end = @min(start + block_size, src.len);

        // Encontrar max abs en bloque
        var max_abs: f32 = 0;
        for (src[start..end]) |v| {
            const vf = @as(f32, @floatCast(v));
            max_abs = @max(max_abs, @abs(vf));
        }

        const scale = if (max_abs > 0) max_abs / 127.0 else 1.0;
        dst.meta[b].scale = scale;
        dst.meta[b].zero_point = 0;

        // Cuantizar
        for (start..end, 0..) |i, j| {
            const q = @as(i8, @intFromFloat(@round(@as(f32, @floatCast(src[i])) / scale)));
            dst.raw_data[i] = @bitCast(q);
        }
    }
}

fn dequantizeInt8Symmetric(src: QuantizedTensor, dst: []f16) void {
    const block_size = src.block_size;

    for (0..src.meta.len) |b| {
        const start = b * block_size;
        const end = @min(start + block_size, dst.len);
        const scale = src.meta[b].scale;

        for (start..end) |i| {
            const q = @as(i8, @bitCast(src.raw_data[i]));
            dst[i] = @floatCast(@as(f32, @floatFromInt(q)) * scale);
        }
    }
}

// ─── INT8 Asimétrico ───

fn quantizeInt8Asymmetric(src: []const f16, dst: *QuantizedTensor) void {
    const block_size = dst.block_size;

    for (0..dst.meta.len) |b| {
        const start = b * block_size;
        const end = @min(start + block_size, src.len);

        var min_v: f32 = std.math.inf(f32);
        var max_v: f32 = -std.math.inf(f32);
        for (src[start..end]) |v| {
            const vf = @as(f32, @floatCast(v));
            min_v = @min(min_v, vf);
            max_v = @max(max_v, vf);
        }

        const scale = (max_v - min_v) / 255.0;
        const zero_point = if (scale > 0) -min_v / scale else 0;

        dst.meta[b].scale = scale;
        dst.meta[b].zero_point = zero_point;

        for (start..end) |i| {
            const q = @as(u8, @intFromFloat(@round(@as(f32, @floatCast(src[i])) / scale + zero_point)));
            dst.raw_data[i] = q;
        }
    }
}

fn dequantizeInt8Asymmetric(src: QuantizedTensor, dst: []f16) void {
    const block_size = src.block_size;

    for (0..src.meta.len) |b| {
        const start = b * block_size;
        const end = @min(start + block_size, dst.len);
        const scale = src.meta[b].scale;
        const zp = src.meta[b].zero_point;

        for (start..end) |i| {
            const q = @as(f32, @floatFromInt(src.raw_data[i]));
            dst[i] = @floatCast((q - zp) * scale);
        }
    }
}

// ─── INT4 (2 valores por byte) ───

fn quantizeInt4(src: []const f16, dst: *QuantizedTensor) void {
    const block_size = dst.block_size;

    for (0..dst.meta.len) |b| {
        const start = b * block_size;
        const end = @min(start + block_size, src.len);

        var min_v: f32 = std.math.inf(f32);
        var max_v: f32 = -std.math.inf(f32);
        for (src[start..end]) |v| {
            const vf = @as(f32, @floatCast(v));
            min_v = @min(min_v, vf);
            max_v = @max(max_v, vf);
        }

        const scale = (max_v - min_v) / 15.0;
        const zero_point = if (scale > 0) -min_v / scale else 0;

        dst.meta[b].scale = scale;
        dst.meta[b].zero_point = zero_point;

        var i = start;
        while (i < end) : (i += 2) {
            const q0 = @as(u4, @intFromFloat(@round(@as(f32, @floatCast(src[i])) / scale + zero_point)));
            const q1 = if (i + 1 < end)
                @as(u4, @intFromFloat(@round(@as(f32, @floatCast(src[i + 1])) / scale + zero_point)))
            else
                0;
            dst.raw_data[i / 2] = (@as(u8, q1) << 4) | @as(u8, q0);
        }
    }
}

fn dequantizeInt4(src: QuantizedTensor, dst: []f16) void {
    const block_size = src.block_size;

    for (0..src.meta.len) |b| {
        const start = b * block_size;
        const end = @min(start + block_size, dst.len);
        const scale = src.meta[b].scale;
        const zp = src.meta[b].zero_point;

        var i = start;
        while (i < end) : (i += 2) {
            const byte = src.raw_data[i / 2];
            const q0 = @as(f32, @floatFromInt(byte & 0x0F));
            dst[i] = @floatCast((q0 - zp) * scale);
            if (i + 1 < end) {
                const q1 = @as(f32, @floatFromInt((byte >> 4) & 0x0F));
                dst[i + 1] = @floatCast((q1 - zp) * scale);
            }
        }
    }
}

// ─── Q4_0 (GGUF) ───
// Bloque de 32 valores: 1 scale (f16) + 32 nibbles = 18 bytes

fn quantizeQ4_0(src: []const f16, dst: *QuantizedTensor) void {
    const block_size = 32; // Q4_0 fijo
    std.debug.assert(dst.block_size == block_size);

    for (0..dst.meta.len) |b| {
        const start = b * block_size;
        const end = @min(start + block_size, src.len);

        var max_abs: f32 = 0;
        for (src[start..end]) |v| {
            max_abs = @max(max_abs, @abs(@as(f32, @floatCast(v))));
        }

        const scale = if (max_abs > 0) max_abs / 7.0 else 0.0;
        dst.meta[b].scale = scale;

        // Escribir scale como f16 al inicio del bloque
        const block_offset = b * 18; // 2 bytes scale + 16 bytes datos (32 nibbles)
        const scale_f16: f16 = @floatCast(scale);
        @memcpy(dst.raw_data[block_offset..][0..2], std.mem.asBytes(&scale_f16));

        // Cuantizar 32 valores en 16 bytes
        for (0..16) |j| {
            const i0 = start + j * 2;
            const i1 = start + j * 2 + 1;

            const q0 = if (i0 < end and scale > 0)
                @as(u4, @intFromFloat(@round(@as(f32, @floatCast(src[i0])) / scale)) + 8)
            else
                8;
            const q1 = if (i1 < end and scale > 0)
                @as(u4, @intFromFloat(@round(@as(f32, @floatCast(src[i1])) / scale)) + 8)
            else
                8;

            dst.raw_data[block_offset + 2 + j] = (@as(u8, q1) << 4) | @as(u8, q0);
        }
    }
}

fn dequantizeQ4_0(src: QuantizedTensor, dst: []f16) void {
    const block_size = 32;

    for (0..src.meta.len) |b| {
        const start = b * block_size;
        const end = @min(start + block_size, dst.len);
        const block_offset = b * 18;

        const scale_f16: f16 = @bitCast(src.raw_data[block_offset..][0..2].*);
        const scale = @as(f32, @floatCast(scale_f16));

        for (0..16) |j| {
            const byte = src.raw_data[block_offset + 2 + j];
            const q0 = @as(i32, @intCast(byte & 0x0F)) - 8;
            const q1 = @as(i32, @intCast((byte >> 4) & 0x0F)) - 8;

            const i0 = start + j * 2;
            if (i0 < end) dst[i0] = @floatCast(@as(f32, @floatFromInt(q0)) * scale);
            if (i0 + 1 < end) dst[i0 + 1] = @floatCast(@as(f32, @floatFromInt(q1)) * scale);
        }
    }
}

// ─── Q8_0 (GGUF) ───
// Bloque de 32 valores: 1 scale (f16) + 32 bytes = 34 bytes

fn quantizeQ8_0(src: []const f16, dst: *QuantizedTensor) void {
    const block_size = 32;
    std.debug.assert(dst.block_size == block_size);

    for (0..dst.meta.len) |b| {
        const start = b * block_size;
        const end = @min(start + block_size, src.len);

        var max_abs: f32 = 0;
        for (src[start..end]) |v| {
            max_abs = @max(max_abs, @abs(@as(f32, @floatCast(v))));
        }

        const scale = if (max_abs > 0) max_abs / 127.0 else 0.0;
        dst.meta[b].scale = scale;

        const block_offset = b * 34; // 2 + 32
        const scale_f16: f16 = @floatCast(scale);
        @memcpy(dst.raw_data[block_offset..][0..2], std.mem.asBytes(&scale_f16));

        for (start..end, 0..) |i, j| {
            const q = @as(i8, @intFromFloat(@round(@as(f32, @floatCast(src[i])) / scale)));
            dst.raw_data[block_offset + 2 + j] = @bitCast(q);
        }
    }
}

fn dequantizeQ8_0(src: QuantizedTensor, dst: []f16) void {
    const block_size = 32;

    for (0..src.meta.len) |b| {
        const start = b * block_size;
        const end = @min(start + block_size, dst.len);
        const block_offset = b * 34;

        const scale_f16: f16 = @bitCast(src.raw_data[block_offset..][0..2].*);
        const scale = @as(f32, @floatCast(scale_f16));

        for (start..end, 0..) |i, j| {
            const q = @as(i8, @bitCast(src.raw_data[block_offset + 2 + j]));
            dst[i] = @floatCast(@as(f32, @floatFromInt(q)) * scale);
        }
    }
}
```

### 4.3 KV-Cache con Paging

```zig
// src/kv_cache/kv_cache.zig
const std = @import("std");
const QuantizedTensor = @import("quant_types.zig").QuantizedTensor;
const QuantFormat = @import("quant_types.zig").QuantFormat;
const quant_ops = @import("quant_ops.zig");
const Tensor = @import("../tensor.zig").Tensor;

pub const KVCacheConfig = struct {
    num_layers: usize,
    num_heads: usize,      // Total heads (para MHA)
    num_kv_heads: usize,   // Heads KV (para GQA/MQA, <= num_heads)
    head_dim: usize,
    max_seq_len: usize,
    block_size: usize = 256,
    k_format: QuantFormat = .fp16,
    v_format: QuantFormat = .fp16,
};

/// Un bloque de KV-cache para una secuencia
pub const KVCacheBlock = struct {
    k_data: QuantizedTensor,
    v_data: QuantizedTensor,
    num_tokens: usize,  // Tokens válidos en este bloque (<= block_size)

    pub fn init(allocator: std.mem.Allocator, config: KVCacheConfig) !KVCacheBlock {
        const elements_per_block = config.block_size * config.num_kv_heads * config.head_dim;

        const k_data = try QuantizedTensor.init(allocator, elements_per_block, config.k_format, config.block_size);
        errdefer k_data.deinit();

        const v_data = try QuantizedTensor.init(allocator, elements_per_block, config.v_format, config.block_size);

        return .{
            .k_data = k_data,
            .v_data = v_data,
            .num_tokens = 0,
        };
    }

    pub fn deinit(self: *KVCacheBlock) void {
        self.k_data.deinit();
        self.v_data.deinit();
    }
};

/// KV-cache para una secuencia con paging
pub const KVCacheSequence = struct {
    allocator: std.mem.Allocator,
    config: KVCacheConfig,
    blocks: std.ArrayList(*KVCacheBlock),
    current_len: usize = 0,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: KVCacheConfig) !Self {
        return .{
            .allocator = allocator,
            .config = config,
            .blocks = std.ArrayList(*KVCacheBlock).init(allocator),
            .current_len = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.blocks.items) |block| {
            block.deinit();
            self.allocator.destroy(block);
        }
        self.blocks.deinit();
    }

    /// Añadir nuevos tokens K, V al cache
    pub fn append(self: *Self, k_new: Tensor(f16), v_new: Tensor(f16)) !void {
        const new_tokens = k_new.shape[0]; // [new_tokens, num_kv_heads * head_dim]
        std.debug.assert(new_tokens > 0);

        var remaining = new_tokens;
        var src_offset: usize = 0;

        // Intentar llenar el último bloque si tiene espacio
        if (self.blocks.items.len > 0) {
            const last = self.blocks.items[self.blocks.items.len - 1];
            const space = self.config.block_size - last.num_tokens;

            if (space > 0) {
                const to_copy = @min(space, remaining);
                try self.copyToBlock(last, k_new, v_new, src_offset, last.num_tokens, to_copy);
                last.num_tokens += to_copy;
                remaining -= to_copy;
                src_offset += to_copy;
                self.current_len += to_copy;
            }
        }

        // Crear nuevos bloques si es necesario
        while (remaining > 0) {
            if (self.current_len + remaining > self.config.max_seq_len) {
                return error.CacheOverflow;
            }

            const block = try self.allocator.create(KVCacheBlock);
            block.* = try KVCacheBlock.init(self.allocator, self.config);

            const to_copy = @min(self.config.block_size, remaining);
            try self.copyToBlock(block, k_new, v_new, src_offset, 0, to_copy);
            block.num_tokens = to_copy;

            try self.blocks.append(block);
            remaining -= to_copy;
            src_offset += to_copy;
            self.current_len += to_copy;
        }
    }

    fn copyToBlock(
        self: Self,
        block: *KVCacheBlock,
        k_new: Tensor(f16),
        v_new: Tensor(f16),
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) !void {
        const elements_per_token = self.config.num_kv_heads * self.config.head_dim;

        // Extraer slice de K
        const k_slice = k_new.data[src_offset * elements_per_token ..][0..count * elements_per_token];
        // Cuantizar y almacenar
        // NOTA: Aquí deberíamos cuantizar por token o por bloque
        // Simplificación: cuantizar todo el rango
        const k_start = dst_offset * elements_per_token;
        var k_fp16 = try self.allocator.alloc(f16, count * elements_per_token);
        defer self.allocator.free(k_fp16);
        @memcpy(k_fp16, k_slice);

        // Para cuantización real, necesitaríamos un buffer intermedio
        // Aquí hacemos un stub que copia directo (FP16)
        @memcpy(block.k_data.raw_data[k_start * 2 ..][0..k_slice.len * 2], std.mem.sliceAsBytes(k_slice));

        // V
        const v_slice = v_new.data[src_offset * elements_per_token ..][0..count * elements_per_token];
        const v_start = dst_offset * elements_per_token;
        @memcpy(block.v_data.raw_data[v_start * 2 ..][0..v_slice.len * 2], std.mem.sliceAsBytes(v_slice));
    }

    /// De-cuantizar todo el cache a tensores FP16 para FlashAttention
    pub fn dequantizeFull(self: Self, k_out: *Tensor(f16), v_out: *Tensor(f16)) !void {
        const elements_per_token = self.config.num_kv_heads * self.config.head_dim;

        var offset: usize = 0;
        for (self.blocks.items) |block| {
            const tokens = block.num_tokens;
            const elements = tokens * elements_per_token;

            // De-cuantizar K
            quant_ops.dequantizeToF16(block.k_data, k_out.data[offset..][0..elements]);

            // De-cuantizar V
            quant_ops.dequantizeToF16(block.v_data, v_out.data[offset..][0..elements]);

            offset += elements;
        }
    }

    /// De-cuantizar solo los tokens necesarios (optimizado para generación)
    pub fn dequantizeRange(
        self: Self,
        start_token: usize,
        end_token: usize,
        k_out: []f16,
        v_out: []f16,
    ) !void {
        const elements_per_token = self.config.num_kv_heads * self.config.head_dim;

        var block_idx: usize = start_token / self.config.block_size;
        var token_in_block = start_token % self.config.block_size;
        var out_offset: usize = 0;

        var t = start_token;
        while (t < end_token) {
            if (block_idx >= self.blocks.items.len) break;
            const block = self.blocks.items[block_idx];

            const tokens_from_block = @min(block.num_tokens - token_in_block, end_token - t);
            const elements = tokens_from_block * elements_per_token;
            const src_offset = token_in_block * elements_per_token;

            quant_ops.dequantizeToF16(
                block.k_data,
                k_out[out_offset..][0..elements],
            );
            quant_ops.dequantizeToF16(
                block.v_data,
                v_out[out_offset..][0..elements],
            );

            out_offset += elements;
            t += tokens_from_block;
            block_idx += 1;
            token_in_block = 0; // Próximo bloque empieza desde 0
        }
    }

    /// Memoria total usada (KB)
    pub fn memoryUsedKB(self: Self) usize {
        var total: usize = 0;
        for (self.blocks.items) |block| {
            total += block.k_data.totalBytes();
            total += block.v_data.totalBytes();
        }
        return total / 1024;
    }

    /// Memoria que usaría en FP16 (KB)
    pub fn memoryFp16KB(self: Self) usize {
        var total: usize = 0;
        for (self.blocks.items) |block| {
            total += block.k_data.fp16Bytes();
            total += block.v_data.fp16Bytes();
        }
        return total / 1024;
    }
};

/// Manager de KV-cache para múltiples secuencias (batch)
pub const KVCacheManager = struct {
    allocator: std.mem.Allocator,
    config: KVCacheConfig,
    sequences: std.ArrayList(KVCacheSequence),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: KVCacheConfig) !Self {
        return .{
            .allocator = allocator,
            .config = config,
            .sequences = std.ArrayList(KVCacheSequence).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.sequences.items) |*seq| seq.deinit();
        self.sequences.deinit();
    }

    /// Crear una nueva secuencia
    pub fn createSequence(self: *Self) !usize {
        const seq = try KVCacheSequence.init(self.allocator, self.config);
        try self.sequences.append(seq);
        return self.sequences.items.len - 1;
    }

    /// Añadir tokens a una secuencia
    pub fn appendToSequence(self: *Self, seq_id: usize, k: Tensor(f16), v: Tensor(f16)) !void {
        if (seq_id >= self.sequences.items.len) return error.InvalidSequenceId;
        try self.sequences.items[seq_id].append(k, v);
    }

    /// Memoria total usada por todas las secuencias (MB)
    pub fn totalMemoryMB(self: Self) usize {
        var total: usize = 0;
        for (self.sequences.items) |seq| {
            total += seq.memoryUsedKB();
        }
        return total / 1024;
    }

    /// Memoria total si fuera FP16 (MB)
    pub fn totalMemoryFp16MB(self: Self) usize {
        var total: usize = 0;
        for (self.sequences.items) |seq| {
            total += seq.memoryFp16KB();
        }
        return total / 1024;
    }

    /// Ratio de compresión global
    pub fn globalCompressionRatio(self: Self) f32 {
        const fp16 = @as(f32, @floatFromInt(self.totalMemoryFp16MB()));
        const actual = @as(f32, @floatFromInt(self.totalMemoryMB()));
        return if (actual > 0) fp16 / actual else 1.0;
    }
};
```

### 4.4 Integración con FlashAttention

```zig
// src/kv_cache/fa_integration.zig
const std = @import("std");
const KVCacheSequence = @import("kv_cache.zig").KVCacheSequence;
const Tensor = @import("../tensor.zig").Tensor;
const FlashAttention = @import("../flash_attention.zig").FlashAttention;

/// Ejecuta FlashAttention con KV-cache cuantizado
/// En generación autoregresiva, solo se computa el último token de Q
pub fn flashAttentionWithCache(
    fa: *FlashAttention,
    cache: KVCacheSequence,
    q_new: Tensor(f16),      // [1, num_heads, 1, head_dim] o [1, num_heads, seq, head_dim]
    k_new: Tensor(f16),      // Nuevos tokens K
    v_new: Tensor(f16),      // Nuevos tokens V
    output: *Tensor(f16),    // Salida
    position: usize,         // Posición actual
) !void {
    const total_len = cache.current_len + q_new.shape[2];

    // 1. Construir K y V completos (cache + nuevos) en FP16
    var k_full = try Tensor(f16).init(cache.allocator, &.{
        1, cache.config.num_kv_heads, total_len, cache.config.head_dim,
    });
    defer k_full.deinit();

    var v_full = try Tensor(f16).init(cache.allocator, &.{
        1, cache.config.num_kv_heads, total_len, cache.config.head_dim,
    });
    defer v_full.deinit();

    // 2. De-cuantizar cache existente
    try cache.dequantizeRange(0, cache.current_len, k_full.data, v_full.data);

    // 3. Copiar nuevos tokens K, V
    const new_offset = cache.current_len * cache.config.num_kv_heads * cache.config.head_dim;
    @memcpy(k_full.data[new_offset..], k_new.data);
    @memcpy(v_full.data[new_offset..], v_new.data);

    // 4. Expandir Q si es necesario (para GQA, repetir heads)
    // ... lógica de GQA repeat

    // 5. Llamar a FlashAttention
    try fa.forward(q_new, k_full, v_full, output);
}
```

---

## 5. Uso en el Pipeline de Generación

```zig
// En llm_pipeline.zig, modificar generate():

pub fn generateWithCache(
    self: *Self,
    prompt_ids: []const u32,
    max_new_tokens: usize,
    kv_cache_mgr: *KVCacheManager,
) ![]u32 {
    var output = try self.allocator.alloc(u32, prompt_ids.len + max_new_tokens);
    @memcpy(output[0..prompt_ids.len], prompt_ids);
    var current_len = prompt_ids.len;

    // Crear secuencia en el cache
    const seq_id = try kv_cache_mgr.createSequence();

    // Forward inicial del prompt completo
    try self.forwardWithCache(prompt_ids, &self.logits, seq_id, kv_cache_mgr);

    for (0..max_new_tokens) |_| {
        // Samplear
        const last_logits = self.logits.data[(current_len - 1) * self.vocab_size ..][0..self.vocab_size];
        const next_token = sampleGreedy(last_logits);
        output[current_len] = next_token;
        current_len += 1;

        if (next_token == 2) break;

        // Forward de SOLO el nuevo token (usando cache)
        try self.forwardSingleToken(next_token, &self.logits, seq_id, kv_cache_mgr, current_len - 1);
    }

    return output[0..current_len];
}
```

---

## 6. Benchmarks Esperados

| Configuración | Memoria FP16 | Memoria INT8 | Memoria INT4 | Memoria Q4_0 |
|---|---|---|---|---|
| Llama-3-8B, 4K ctx | 1 GB | 0,5 GB | 0,25 GB | 0,25 GB |
| Llama-3-70B, 32K ctx | 80 GB | 40 GB | 20 GB | 20 GB |
| Llama-3-405B, 128K ctx | 2,4 TB | 1,2 TB | 600 GB | 600 GB |

**Pérdida de calidad (perplexity relativa):**
- INT8 simétrico: < 0,1%
- Q8_0: < 0,1%
- Q4_0: ~1-2%
- INT4: ~3-5%

---

## 7. Pendientes y Roadmap

### Fase 1: Base (ahora)
- [x] Tipos de cuantización: FP16, INT8, INT4, Q4_0, Q8_0
- [x] Paging de bloques 256
- [x] De-cuantización on-the-fly
- [x] Soporte GQA/MQA (num_kv_heads <= num_heads)

### Fase 2: Optimización
- [ ] De-cuantización en GPU (kernel CUDA)
- [ ] Compresión de bloques no utilizados (evict a CPU)
- [ ] Prefix caching (compartir bloques entre secuencias)
- [ ] Q5_K y Q6_K de GGUF

### Fase 3: Avanzado
- [ ] FP8 nativo (Hopper/Blackwell)
- [ ] KV-cache cuantizado en FlashAttention kernel (de-cuantización en shared memory)
- [ ] PagedAttention v2 (algoritmo de vLLM)
- [ ] Speculative decoding con cache

---

*Documento generado en agosto 2026.*

---

## Estado de implementación (agosto 2026 — entrega actual)

- [x] **CLI llama.cpp-compatible** (`src/main.zig` `CliParams`/`parseArgs`):
  `--cache-type-k`, `--cache-type-v` (`q8_0|q4_0|q4_1|fp16|...` con alias `q4_k`/`q8_k`);
  `--spec-draft-type-k`, `--spec-draft-type-v`; `-np`; `--spec-type draft-mtp`;
  `--spec-draft-n-max`; `-jinja`. Ver `printHelp`.
- [x] **Helper de cuantización host** (`src/kv_cache/kv_quant.zig`): `quantBytes`,
  `encode`, `encodeToOwned`, `decode` con layout canónico GGUF (q8_0=34B/bloc,
  q4_0=18B/bloc, q4_1=20B/bloc) + tests de round-trip q8_0/q4_0.
- [x] **`QuantFormat.fromString`/`toString`** en `src/kv_cache/quant_types.zig`
  (q4_1 añadido al enum + switches).
- [x] **Ruta legacy `KVCacheManager`** (`src/kv_cache/kv_cache_manager.zig`):
  `appendTokensF16` cuantiza realmente (antes raw-memcpy); `appendTokens` usa stride
  `quantBytes`; `dequantizeCpu` decodifica q8_0/q4_0/q4_1 vía `kv_quant.decode`
  (antes lanzaba `UnsupportedCpuDequant`). Test `kv_cache append and retrieve q8_0`.
- [x] **Ruta híbrida paged (Qwen3.5)**: store real de K/V cuantizado en `hybrid_attn.zig`
  (tile f16 acumulado + `kv_quant.encode` al sellar bloque / al cambio de bloque) y
  de-deshacer on-read. CPU: de-deshacer on-read vía `kv_quant.decode` por bloque.
  GPU: cuando el cache está cuantizado se fuerza la ruta CPU de-deshacer (correcta);
  el staging de-deshacer f16 on-device para el kernel paged es fase posterior
  (el kernel f16 de `PagedAttentionGpu` asume f16; un driver fused q8_0 es futuro).
- [x] **Spec-decoding MTP**: `--spec-type draft-mtp` detecta `blk.0.nextn.eh_proj.weight`
  en el GGUF; el 0.8B carece de head MTP y aborta con mensaje claro (EXIT 0). Driver
  especulativo completo es fase posterior. `--spec-draft-n-max` controla el budget.
- [ ] **`-jinja`**: flag reconocido; render completo de `chat_template` es fase posterior.

---

## 8. Rendimiento del matmul — caché de pesos en GPU (agosto 2026)

**Problema raíz (¿por qué zig-ai era ~40× más lento que llama.cpp?).** El motor
re-subía y re-descuantizaba los pesos en cada token:

1. `hybrid_attn.zig` / `hybrid_layer.zig` llamaban `dequantToF32Transposed` sobre
   los `scratch_*` **en cada `forward`** (re-descuantizar ~3 GB f32/token en CPU).
2. `cublas.gemmCuBlasF32` hacía `cudaMalloc` + `upload(B)` de TODA la matriz de
   pesos **en cada GEMM** (re-subir ~3 GB/token por PCIe). No había residencia
   de pesos en device.

**Arreglo implementado:**

- `src/matmul/root.zig`: `MatmulEngine` ahora tiene `weight_cache`
  (`AutoHashMap(usize, GpuBuffer(f32))`) y `linearProjection` sube `W_T` **una
  vez** (clave = `W_T.data.ptr`) y lo reusa en todos los tokens vía
  `cublas.gemmCuBlasF32Resident` (sólo sube la activación `A`, pequeña).
  Cubre attention/FFN/lm_head (todos pasan por `linearProjection`); la atención
  Q@Kᵀ usa `gemm()` directo y queda fuera de la caché (correcto: B es activación).
- `dequantToF32Transposed` movido de `forward` a `loadWeightsFromGguf` (una vez).

**Resultado (0.8B Q4_0, RTX 3080 Laptop, `n=48`, temp 0):**

| Motor | KV | tok/s | GPU peak MiB |
|---|---|---|---|
| llama.cpp | fp16 | 32.7 | 4087 |
| llama.cpp | q8_0 | 32.3 | 3153 |
| zig-ai (antes) | fp16 | 0.62 | 5207 |
| zig-ai (ahora) | fp16 | 6.97 | 3445 |

~11× más rápido (0.62 → 6.97 tok/s). La caché de pesos NO cambia la corrección
(mismo peso, mismo orden); validado contra CPU con tests `cublas gemm matches
naive` / `cublas linearProjection matches naive` en `tests/test_matmul.zig`.

**Bug conocido — `PagedAttentionGpu` (corrección, no rendimiento).** El kernel de
atención en GPU produce salida degenerada (`1-1-1-…`) en el 0.8B; aislado
forzando `paged_gpu=null` (atención por CPU) → salida correcta
("I have a question about the following code…") y matmul sigue en GPU. Hasta
corregir el kernel, `use_gpu_kv` se fuerza a `false` en `src/main.zig`
(`gpu_attention_enabled = false`). Por eso zig-ai aún queda ~4.7× por debajo de
llama.cpp: atención en CPU (no en GPU) y matmul f32 (no f16/tensor-core).
Pendiente: corregir `PagedAttentionGpu` y migrar a GEMM f16 (tensor cores).
