//! KVarN: cuantización variance-normalized del KV-cache (BeeLlama port)
//!
//! Layout de records + descriptores + API de codificación/decodificación
//! CPU reference para el ladder de cache comprimido del puerto BeeLlama.
//!
//! ## Contrato C1 (B2 → B1)
//!
//! `KvarnRecordLayout` define la forma canónica del record KVarN en memoria:
//!
//! - Grupo fijo de `KVAR_N_GROUP = 128` tokens por record (constraint del
//!   algoritmo Sinkhorn).
//! - Bit-widths K y V independientes (`key_bits`, `value_bits` ∈ {2,3,4,5,6,8}).
//! - Cada record contiene payload K comprimido + escalas + zero-points +
//!   payload V comprimido + escalas + zero-points (layout `llama_kvarn_tile_layout`).
//! - **Alineación**: cada record se redondea a múltiplos de 32 bytes (sector
//!   L2). Esto evita el bug lane-b de "sectores compartidos ⇒ write-combining
//!   L2 pierde datos" documentado en `HANDOFFS.md`. Overhead de padding ~5-15%
//!   en reposo, pero libera a B1 de serialización obligatoria.
//! - **Sufijo exacto intrínseco**: ring buffer separado de 128 slots f16
//!   (`KvarnExactRing`), rotado en cada append. La FA nativa combina cuerpo
//!   comprimido + cola exacta sin materializar el cuerpo.
//!
//! ## Algoritmo (transcrito de beellama.cpp/src/llama-kvarn.cpp)
//!
//! 1. **Rotación Hadamard 128** sobre el tile fp32 (`hadamard128InPlace`,
//!    matriz 128×128 Hadamard normalizada por 1/√128, butterfly Cooley-Tukey).
//! 2. **Normalización Sinkhorn** iterativa: equilibra std por filas/columnas
//!    del tile rotado, trackeando `s_col[i]` y `s_row[j]` (fp32, log-space con
//!    clamps ±0.3/10.0). Default `sinkhorn_iters = 16`. Solo se conservan las
//!    escalas que minimizan la "imbalance" (ratio max/min std fila+columna).
//! 3. **Cuantización por filas**: `q = round((balanced[r,c] - lo) / scale)`
//!    donde `lo = min fila`, `scale = (hi-lo)/qmax`. `qmax = (1<<bits)-1`.
//! 4. **Almacenamiento de escalas (fp16)**:
//!    - K: `s_col_fp16[c]` (column scale), `zp_row_fp16[r] = s_row*lo`,
//!      `s_row_fp16[r] = s_row*scale`.
//!    - V: `s_row_fp16[r] = s_row*scale`, `zp_row_fp16[r] = s_row*lo`,
//!      `s_col_fp16[c]`.
//! 5. **Dequantización**: `tile[r,c] = (q * scale_row + zp_row) * scale_col`.
//!
//! ## Uso (CPU reference, sin GPU)
//!
//! ```zig
//! const kvarn = @import("kvarn.zig");
//! const qt = @import("quant_types.zig");
//!
//! var record: [kvarn.recordBytes(128, 4, 4)]u8 = undefined;
//! var tile: [128 * 128]f32 = undefined;
//! // ... fill tile with fp32 K or V values ...
//! kvarn.encodeKTile(&tile, 16, 4, &record);
//! // ... pass to FA ...
//! var reconstructed: [128 * 128]f32 = undefined;
//! kvarn.decodeKTile(&record, 4, &reconstructed);
//! ```

const std = @import("std");

/// Tamaño de grupo KVarN en tokens (constraint del algoritmo Sinkhorn).
pub const KVAR_N_GROUP: u32 = 128;

/// Tamaño de bloque KVarN en elementos lógicos (group × head_dim / slices).
pub const KVAR_N_BLOCK: usize = KVAR_N_GROUP;

/// Sectores L2 asumidos. Lección lane-b: stores a records no alineados a 32B
/// ⇒ sectores compartidos ⇒ write-combining L2 PIERDE datos ⇒ __syncwarp +
/// __stcg obligatorios. C1 freeze: records padded a múltiplos de 32B → B1
/// puede usar stores coalesced estándar sin serialización.
pub const KVAR_L2_SECTOR: usize = 32;

/// Bits válidos para KVarN. El encoder/decoder valida `key_bits`/`value_bits`
/// contra este set; cualquier otro valor produce `error.UnsupportedKvarBits`.
pub const valid_bits = [_]u8{ 2, 3, 4, 5, 6, 8 };

/// Descriptor de un tipo KVarN (par K_bits × V_bits).
///
/// Equivalente a `llama_kvarn_type_desc` en C++. Zig usa un struct de comptime
/// data (no enum) porque el espacio (KBits, VBits) ∈ {2,3,4,5,6,8}² no encaja
/// en una enumeración finita razonable y queremos un único constructor.
pub const KvarnType = struct {
    key_bits: u8,
    value_bits: u8,

    /// Valida que el par de bit-widths es soportado.
    pub fn isValid(self: KvarnType) bool {
        return isValidBits(self.key_bits) and isValidBits(self.value_bits);
    }

    /// Nombre canónico estilo BeeLlama: `kvarn_k<N>v<M>_g128`.
    /// Retorna un slice a un buffer estático compartido (no usar entre
    /// llamadas sin copiar primero).
    pub fn name(self: KvarnType) []const u8 {
        // Buffer estático thread-local para que el slice siga válido tras
        // retornar. Tamaño máximo: "kvarn_k8v8_g128" = 15 chars + NUL.
        const Static = struct {
            var buf: [16:0]u8 = [1:0]u8{0} ** 16;
        };
        const slice = std.fmt.bufPrintZ(&Static.buf, "kvarn_k{d}v{d}_g128", .{ self.key_bits, self.value_bits }) catch unreachable;
        return slice[0..slice.len];
    }

    /// Parsea `kvarn_k<N>v<M>_g128`. Devuelve `null` si el formato no encaja.
    pub fn parse(s: []const u8) ?KvarnType {
        const prefix = "kvarn_k";
        if (!std.mem.startsWith(u8, s, prefix)) return null;
        var rest = s[prefix.len..];
        // Parsea primer número (K bits).
        const k_end = std.mem.indexOfAny(u8, rest, "v") orelse return null;
        const k_str = rest[0..k_end];
        rest = rest[k_end + 1 ..];
        // Parsea segundo número (V bits).
        const v_end = std.mem.indexOfAny(u8, rest, "_") orelse return null;
        const v_str = rest[0..v_end];
        rest = rest[v_end..];
        if (!std.mem.eql(u8, rest, "_g128")) return null;
        const k_bits = std.fmt.parseInt(u8, k_str, 10) catch return null;
        const v_bits = std.fmt.parseInt(u8, v_str, 10) catch return null;
        const result = KvarnType{ .key_bits = k_bits, .value_bits = v_bits };
        return if (result.isValid()) result else null;
    }
};

/// `true` si el bit-width es uno de los valores KVarN soportados.
pub fn isValidBits(bits: u8) bool {
    for (valid_bits) |b| if (b == bits) return true;
    return false;
}

/// Layout de un record KVarN dentro de un slot del pool.
///
/// Equivalente a `llama_kvarn_tile_layout` (C++). Estructura del record:
///
/// ```text
/// | k_payload  | k_s_col (head_dim × f16) | k_zp (head_dim × f16) |
/// | k_s_row (group × f16) | v_payload | v_s_col (head_dim × f16) |
/// | v_s_row (group × f16) | v_zp (group × f16) | <padding → 32B> |
/// ```
///
/// Importante para B1: `tile_bytes` siempre es múltiplo de `KVAR_L2_SECTOR`.
pub const KvarnRecordLayout = struct {
    /// Offset del payload K comprimido (packed bits: `group × head_dim × bits`).
    k_payload_off: usize,
    /// Bytes del payload K.
    k_payload_bytes: usize,
    /// Offset de las escalas column-wise de K (`head_dim × u16` fp16).
    k_s_col_off: usize,
    /// Offset de los zero-points column-wise de K (`head_dim × u16` fp16).
    k_zp_off: usize,
    /// Offset de las escalas row-wise de K (`group × u16` fp16).
    k_s_row_off: usize,

    /// Offset del payload V comprimido (packed bits: `group × head_dim × bits`).
    v_payload_off: usize,
    /// Bytes del payload V.
    v_payload_bytes: usize,
    /// Offset de las escalas column-wise de V (`head_dim × u16` fp16).
    v_s_col_off: usize,
    /// Offset de las escalas row-wise de V (`group × u16` fp16).
    v_s_row_off: usize,
    /// Offset de los zero-points row-wise de V (`group × u16` fp16).
    v_zp_off: usize,

    /// Bytes totales del record, padded a `KVAR_L2_SECTOR`.
    tile_bytes: usize,

    /// Dimensión de cabeza (128, 256 o 512 en modelos soportados).
    head_dim: u32,
    /// Tamaño de grupo (siempre 128).
    group: u32,
    /// Bit-width del payload K.
    key_bits: u8,
    /// Bit-width del payload V.
    value_bits: u8,

    /// Construye el layout para un par `(key_bits, value_bits)` y `head_dim`
    /// dados. El group es fijo (`KVAR_N_GROUP`).
    pub fn init(head_dim: u32, key_bits: u8, value_bits: u8) !KvarnRecordLayout {
        // 9.12 (lane-b) F1: D=64 admitido (tile rect 64×128 — beellama
        // @e1f6d6fe6). El layout C1 por física es el mismo patrón con
        // hd=64 (k_payload 128·64 packed a k_bits).
        if (head_dim != 64 and head_dim != 128 and head_dim != 256 and head_dim != 512) {
            return error.UnsupportedHeadDim;
        }
        if (!isValidBits(key_bits) or !isValidBits(value_bits)) {
            return error.UnsupportedKvarBits;
        }
        const hd: usize = head_dim;
        const group: usize = KVAR_N_GROUP;
        var off: usize = 0;

        const k_payload_bytes = packedBytes(group * hd, key_bits);
        const k_payload_off = off;
        off += k_payload_bytes;

        const k_s_col_off = off;
        off += hd * @sizeOf(u16);

        const k_zp_off = off;
        off += hd * @sizeOf(u16);

        const k_s_row_off = off;
        off += group * @sizeOf(u16);

        const v_payload_bytes = packedBytes(group * hd, value_bits);
        const v_payload_off = off;
        off += v_payload_bytes;

        const v_s_col_off = off;
        off += hd * @sizeOf(u16);

        const v_s_row_off = off;
        off += group * @sizeOf(u16);

        const v_zp_off = off;
        off += group * @sizeOf(u16);

        // C1 freeze: padding a múltiplos de 32B (sector L2) para evitar
        // write-combining races en stores GPU. Ver HANDOFFS.md (lane-b lección).
        const tile_bytes = alignUp(off, KVAR_L2_SECTOR);

        return .{
            .k_payload_off = k_payload_off,
            .k_payload_bytes = k_payload_bytes,
            .k_s_col_off = k_s_col_off,
            .k_zp_off = k_zp_off,
            .k_s_row_off = k_s_row_off,
            .v_payload_off = v_payload_off,
            .v_payload_bytes = v_payload_bytes,
            .v_s_col_off = v_s_col_off,
            .v_s_row_off = v_s_row_off,
            .v_zp_off = v_zp_off,
            .tile_bytes = tile_bytes,
            .head_dim = head_dim,
            .group = KVAR_N_GROUP,
            .key_bits = key_bits,
            .value_bits = value_bits,
        };
    }
};

/// Bytes necesarios para empaquetar `n_values` elementos a `bits` por elemento.
pub fn packedBytes(n_values: usize, bits: u8) usize {
    return (n_values * bits + 7) / 8;
}

/// Bytes totales de un record para `head_dim`, `key_bits`, `value_bits`.
/// Wrapper sobre `KvarnRecordLayout.init`.
pub fn recordBytes(head_dim: u32, key_bits: u8, value_bits: u8) usize {
    return (KvarnRecordLayout.init(head_dim, key_bits, value_bits) catch unreachable).tile_bytes;
}

/// Alinea `value` hacia arriba al próximo múltiplo de `alignment`.
fn alignUp(value: usize, alignment: usize) usize {
    return (value + alignment - 1) / alignment * alignment;
}

/// Ring buffer de slots f16 exactos para el sufijo intrínseco de 128 tokens.
///
/// Equivalente al `llama_kv_cache_kvarn::k_tail` / `v_tail` en C++. Ring
/// circular: cada append sobrescribe el slot más antiguo; el número de slots
/// exactos es `exact_groups` (1 por defecto en Bee = solo el grupo vivo).
///
/// Decisión del contract C1: ring separado del cuerpo KVarN comprimido. La
/// FA nativa consume ambos buffers sin materializar el cuerpo. Esto evita
/// el bug lane-b de "sectores compartidos" porque cada ring es un buffer
/// f16 limpio (2 bytes/elem, sin escalas), trivialmente sector-aligned.
pub fn KvarnExactRing(comptime slots: u32) type {
    return struct {
        const Self = @This();
        /// Slots f16 exactos. Layout: `[slots][head_dim]` fp16 (row-major
        /// token-major). En C++ el equivalente es `tensor<group × head_dim>`
        /// rotativo.
        buffer: [*]f16,
        /// Capacidad máxima en filas (slots).
        capacity: u32,
        /// Dimensión de cabeza (cada fila tiene `head_dim` fp16).
        head_dim: u32,
        /// Índice de escritura (próximo slot a sobrescribir). Crece monotono
        /// y se reduce módulo `capacity`.
        write_idx: u64,

        /// Bytes totales del ring (`slots × head_dim × 2`). Útil para reservar
        /// memoria del device o del host.
        pub fn byteSize(head_dim: u32) usize {
            return slots * head_dim * @sizeOf(f16);
        }

        /// Inicializa el ring sobre un slice f16 ya alocado de tamaño
        /// `slots * head_dim`. Para buffers de array fijo, el llamante debe
        /// aplicar `buffer[0..]`.
        pub fn init(buffer: []f16, head_dim: u32) !Self {
            const expected = comptime slots;
            if (buffer.len != expected * head_dim) return error.RingSizeMismatch;
            return .{
                .buffer = buffer.ptr,
                .capacity = slots,
                .head_dim = head_dim,
                .write_idx = 0,
            };
        }

        /// Escribe un token (`head_dim` fp16) en el siguiente slot.
        pub fn writeToken(self: *Self, token: []const f16) void {
            std.debug.assert(token.len == self.head_dim);
            const slot = @as(u32, @intCast(self.write_idx % self.capacity));
            const dst = self.buffer[slot * self.head_dim ..][0..self.head_dim];
            @memcpy(dst, token);
            self.write_idx += 1;
        }

        /// Devuelve puntero de lectura al token escrito en `logical_idx` pasos
        /// atrás (0 = más reciente, `capacity-1` = más antiguo). El llamante
        /// debe respetar la vida del ring y consultar `numValid` antes.
        pub fn tokenAt(self: *const Self, logical_idx: u32) [*]const f16 {
            std.debug.assert(logical_idx < self.capacity);
            const total = @as(u64, self.write_idx);
            const offset = if (total > logical_idx)
                @as(u32, @intCast((total - 1 - logical_idx) % self.capacity))
            else
                0;
            return @as([*]const f16, @ptrCast(&self.buffer[offset * self.head_dim]));
        }

        /// Número de tokens actualmente escritos (saturado a `capacity`).
        pub fn numValid(self: *const Self) u32 {
            return @as(u32, @intCast(@min(self.write_idx, self.capacity)));
        }
    };
}

/// Tipo del ring exacto usado por defecto. 1 slot exacto (intrinsic group):
/// Bee mantiene solo el grupo vivo en f16; grupos completados se quantizan y
/// caen al cuerpo KVarN. 2 slots en non-SWA (rollback insurance); ver
/// `llama_kvarn_non_swa_tail_groups` en C++.
pub const ExactRingDefault = KvarnExactRing(1);

/// Versión con 2 slots para rollback (non-SWA). Útil para tests que simulan
/// la mecánica de rollback.
pub const ExactRingRollback = KvarnExactRing(2);

// ============================================================================
// Algoritmo: Hadamard 128 + Sinkhorn + cuantización
// ============================================================================

/// Aplica la rotación Hadamard 128×128 normalizada IN-PLACE sobre un buffer
/// fp32 de 128 elementos. Equivalente a `llama_kvarn_hadamard_128`.
///
/// Butterfly Cooley-Tukey: para `stride ∈ {1,2,4,8,16,32,64}` aplica
/// `(a+b, a-b)` en pares. Al final multiplica por `1/√128` (normalización
/// unitaria que preserva Parseval).
pub fn hadamard64InPlace(values: *[64]f32) void {
    // 9.12 (lane-b) F1: WHT-64 — 6 etapas butterfly, 1/√64.
    // Espejo de kvarn_wht_64_lane beellama @e1f6d6fe6 (smem
    // d64: 64·128 + 8·128 + 18 escalares).
    var stride: usize = 1;
    while (stride < 64) : (stride *= 2) {
        var base: usize = 0;
        while (base < 64) : (base += 2 * stride) {
            var i: usize = 0;
            while (i < stride) : (i += 1) {
                const a = values[base + i];
                const b = values[base + stride + i];
                values[base + i] = a + b;
                values[base + stride + i] = a - b;
            }
        }
    }
    const inv_sqrt_64: f32 = 0.125;
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        values[i] *= inv_sqrt_64;
    }
}

pub fn hadamard128InPlace(values: *[128]f32) void {
    var stride: usize = 1;
    while (stride < 128) : (stride *= 2) {
        var base: usize = 0;
        while (base < 128) : (base += 2 * stride) {
            var i: usize = 0;
            while (i < stride) : (i += 1) {
                const a = values[base + i];
                const b = values[base + stride + i];
                values[base + i] = a + b;
                values[base + stride + i] = a - b;
            }
        }
    }
    const inv_sqrt_128: f32 = 0.08838834764831845;
    var i: usize = 0;
    while (i < 128) : (i += 1) {
        values[i] *= inv_sqrt_128;
    }
}

/// Aplica la Hadamard 128×128 a las 128 filas de una matriz
/// `[group=128][head_dim=128]` (fila por fila, en sitio).
///
/// Útil para procesar un tile completo sin reservar un buffer scratch.
pub fn hadamard128Rows(tile: []f32, head_dim: usize) void {
    std.debug.assert(tile.len == KVAR_N_GROUP * head_dim);
    var row: [128]f32 = undefined;
    var r: usize = 0;
    while (r < KVAR_N_GROUP) : (r += 1) {
        const off = r * head_dim;
        var c: usize = 0;
        while (c < head_dim) : (c += 1) row[c] = tile[off + c];
        // Hadamard solo sobre los primeros 128 elementos (slices). Si
        // head_dim > 128, las columnas excedentes se copian sin transformar.
        hadamard128InPlace(&row);
        c = 0;
        while (c < head_dim) : (c += 1) tile[off + c] = row[c];
    }
}

/// 9.4 (lane-b): Hadamard por slices — el pipeline beellama D≥256 aplica
/// WHT-128 intra-slice a CADA slice y después el butterfly cross-slice
/// entre slices (kvarn.cu:1086-1105: `kvarn_wht_128` por slice +
/// `kvarn_wht_cross_slices` ×1/√S). `hadamard128Rows` legacy solo rota el
/// slice-0 — INCONSISTENTE con el store GPU; esta variante es la canónica
/// para D=256/512 y coincide bit-conceptualmente con
/// fattn_kvarn_portable_d256_kernel (fattn_kvarn_portable.cu:820-824:
/// wht_128 × slices luego cross-slices).
///
/// La transformación completa es W = (I_S ⊗ H128)·C_S con C_S el
/// butterfly entre slices — ambas ortonormales ⇒ W ortonormal ⇒
/// preserva Parseval y es involutiva (tests abajo).
pub fn hadamardSlicesRows(tile: []f32, head_dim: usize) void {
    std.debug.assert(tile.len == KVAR_N_GROUP * head_dim);
    // 9.12 (lane-b) F1: D=64 ⇒ 1 slice de 64 con WHT-64 (sin cross —
    // beellama d64: kvarn_wht_64_lane, tile rect 64×128).
    if (head_dim == 64) {
        var row: [64]f32 = undefined;
        var r: usize = 0;
        while (r < KVAR_N_GROUP) : (r += 1) {
            const off = r * 64;
            var c: usize = 0;
            while (c < 64) : (c += 1) row[c] = tile[off + c];
            hadamard64InPlace(&row);
            c = 0;
            while (c < 64) : (c += 1) tile[off + c] = row[c];
        }
        return;
    }
    std.debug.assert(head_dim % 128 == 0);
    const slices = head_dim / 128;

    // 1) WHT-128 intra-slice en cada fila (cada slice de 128 columnas).
    // NB: hadamard128Rows solo puede hd=128 (row scratch [128]) — aquí
    // operamos directamente sobre el tile con el mismo butterfly.
    var r: usize = 0;
    while (r < KVAR_N_GROUP) : (r += 1) {
        var s: usize = 0;
        while (s < slices) : (s += 1) {
            var stride: usize = 1;
            const base_off = r * head_dim + s * 128;
            while (stride < 128) : (stride *= 2) {
                var base: usize = 0;
                while (base < 128) : (base += 2 * stride) {
                    var i: usize = 0;
                    while (i < stride) : (i += 1) {
                        const ai = base_off + base + i;
                        const bi = base_off + base + stride + i;
                        const a = tile[ai];
                        const b = tile[bi];
                        tile[ai] = a + b;
                        tile[bi] = a - b;
                    }
                }
            }
        }
    }
    const inv_sqrt_128: f32 = 0.08838834764831845;
    r = 0;
    while (r < KVAR_N_GROUP) : (r += 1) {
        var i: usize = 0;
        while (i < head_dim) : (i += 1) tile[r * head_dim + i] *= inv_sqrt_128;
    }

    // 2) Cross-slice butterfly: para cada fila r y dim d (0..127),
    //    x[s][d] = Σ_t sign butterfly × x[t][d], normalizado ×1/√S.
    //    SLICES=2: (a+b, a-b)·1/√2 — espejo exacto de
    //    fattn_kvarn_wht_cross_slices<SLICES> (portable.cu:728-733).
    var row: [512]f32 = undefined;
    r = 0;
    while (r < KVAR_N_GROUP) : (r += 1) {
        const off = r * head_dim;
        var c: usize = 0;
        while (c < head_dim) : (c += 1) row[c] = tile[off + c];

        var stride: usize = 1;
        while (stride < slices) : (stride *= 2) {
            var base: usize = 0;
            while (base < slices) : (base += 2 * stride) {
                var i: usize = 0;
                while (i < stride) : (i += 1) {
                    var d: usize = 0;
                    while (d < 128) : (d += 1) {
                        const ai = (base + i) * 128 + d;
                        const bi = (base + stride + i) * 128 + d;
                        const a = row[ai];
                        const b = row[bi];
                        row[ai] = a + b;
                        row[bi] = a - b;
                    }
                }
            }
        }
        // Normalización ×1/√S (1/√2 para 2 slices, 1/2 para 4).
        const scale: f32 = switch (slices) {
            1 => 1.0,
            2 => 0.707106781186547524,
            4 => 0.5,
            else => unreachable,
        };
        c = 0;
        while (c < head_dim) : (c += 1) tile[off + c] = row[c] * scale;
    }
}

/// Resultado de la normalización Sinkhorn: tile balanceado + escalas fp32.
pub const SinkhornResult = struct {
    /// Tile balanceado (fila×columna) listo para cuantizar.
    balanced: []f32,
    /// Escalas column-wise (fp32, log-space absorbed).
    s_col: [128]f32,
    /// Escalas row-wise (fp32, log-space absorbed).
    s_row: [128]f32,
};

/// Normalización varianza-Sinkhorn: equilibra std por filas/columnas vía
/// multiplicadores iterativos en log-space. Solo conserva el estado con
/// menor `imbalance` (ratio max/min std fila+columna).
///
/// Equivalente a `llama_kvarn_variance_normalize` en C++.
/// `sinkhorn_iters` debe ser > 0; recomendado 16 (default en Bee).
pub fn varianceNormalize(
    tile: []const f32,
    sinkhorn_iters: u32,
    balanced: []f32,
    s_col: *[128]f32,
    s_row: *[128]f32,
) void {
    std.debug.assert(tile.len == KVAR_N_GROUP * KVAR_N_GROUP);
    std.debug.assert(balanced.len == KVAR_N_GROUP * KVAR_N_GROUP);

    var log_s_col: [128]f32 = [_]f32{0.0} ** 128;
    var log_s_row: [128]f32 = [_]f32{0.0} ** 128;
    @memcpy(balanced, tile);

    @memset(s_col, 1.0);
    @memset(s_row, 1.0);
    var imbalance_best = imbalance(balanced);

    var iter: u32 = 0;
    while (iter < sinkhorn_iters) : (iter += 1) {
        // Column pass: ajustar log_s_col[c] por std de cada columna.
        var c: usize = 0;
        while (c < 128) : (c += 1) {
            var std_col = sampleStd(balanced[c..], 128, 128);
            if (std_col < 1e-3) std_col = 1e-3;
            if (std_col > 1e3) std_col = 1e3;
            // D1 (PLAN_B1): transcendentales deterministas — la secuencia de
            // ops es idéntica en el kernel GPU (kvarn_desc.cuh), garantizando
            // bits idénticos CPU↔GPU sin depender del libm de cada plataforma.
            const new_log = @as(f64, log_s_col[c]) + kvarnLog(@floatCast(std_col));
            log_s_col[c] = @floatCast(clampF64(new_log, -0.3, 10.0));
        }
        rebuildCur(tile, &log_s_col, &log_s_row, balanced);

        // Row pass: ajustar log_s_row[r] por std de cada fila.
        var r: usize = 0;
        while (r < 128) : (r += 1) {
            var std_row = sampleStd(balanced[r * 128 ..][0..128], 128, 1);
            if (std_row < 1e-3) std_row = 1e-3;
            if (std_row > 1e3) std_row = 1e3;
            const new_log = @as(f64, log_s_row[r]) + kvarnLog(@floatCast(std_row));
            log_s_row[r] = @floatCast(clampF64(new_log, -0.3, 10.0));
        }
        rebuildCur(tile, &log_s_col, &log_s_row, balanced);

        const imb = imbalance(balanced);
        if (imb <= imbalance_best) {
            imbalance_best = imb;
            for (s_col, 0..) |*sc, i| sc.* = @floatCast(kvarnExp(log_s_col[i]));
            for (s_row, 0..) |*sr, i| sr.* = @floatCast(kvarnExp(log_s_row[i]));
        }
    }

    // Aplica las mejores escalas al output final.
    var r2: usize = 0;
    while (r2 < 128) : (r2 += 1) {
        var c2: usize = 0;
        while (c2 < 128) : (c2 += 1) {
            balanced[r2 * 128 + c2] = tile[r2 * 128 + c2] / (s_col[c2] * s_row[r2]);
        }
    }
}

fn rebuildCur(
    tile: []const f32,
    log_s_col: *const [128]f32,
    log_s_row: *const [128]f32,
    out: []f32,
) void {
    var r: usize = 0;
    while (r < 128) : (r += 1) {
        const s_row = kvarnExp(log_s_row[r]);
        var c: usize = 0;
        while (c < 128) : (c += 1) {
            out[r * 128 + c] = @floatCast(@as(f64, tile[r * 128 + c]) / (kvarnExp(log_s_col[c]) * s_row));
        }
    }
}

fn clampF64(v: f64, lo: f64, hi: f64) f64 {
    return if (v < lo) lo else if (v > hi) hi else v;
}

fn sampleStd(values: []const f32, n: usize, stride: usize) f32 {
    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const v = values[i * stride];
        sum += v;
        sum_sq += v * v;
    }
    const nf: f64 = @floatFromInt(n);
    const mean = sum / nf;
    const variance = if (n > 1) @max(0.0, (sum_sq - nf * mean * mean) / (nf - 1.0)) else 0.0;
    return @floatCast(@sqrt(variance));
}

fn imbalance(tile: []const f32) f32 {
    var col_min: f32 = std.math.inf(f32);
    var col_max: f32 = 0.0;
    var row_min: f32 = std.math.inf(f32);
    var row_max: f32 = 0.0;

    var c: usize = 0;
    while (c < 128) : (c += 1) {
        const s = sampleStd(tile[c..], 128, 128);
        if (s < col_min) col_min = s;
        if (s > col_max) col_max = s;
    }
    var r: usize = 0;
    while (r < 128) : (r += 1) {
        const s = sampleStd(tile[r * 128 ..][0..128], 128, 1);
        if (s < row_min) row_min = s;
        if (s > row_max) row_max = s;
    }
    const col_min_safe = if (col_min < 1e-8) 1e-8 else col_min;
    const row_min_safe = if (row_min < 1e-8) 1e-8 else row_min;
    return col_max / col_min_safe + row_max / row_min_safe;
}

fn clamp(v: f32, lo: f32, hi: f32) f32 {
    return if (v < lo) lo else if (v > hi) hi else v;
}

// ============================================================================
// Transcendentales deterministas (contrato D1 de PLAN_B1)
// ============================================================================
// `@log`/`@exp` llaman al libm de la plataforma (glibc en CPU, libdevice en
// CUDA) con ulps distintos => imposible bit-exactitud CPU↔GPU. Para que el
// store GPU (lane-b1 A3) sea bit-exacto contra este CPU reference, TODO
// Sinkhorn usa estas variantes con aritmética f64 explícita IDÉNTICA en Zig
// y en el kernel (src/cuda/kvarn_desc.cuh define las mismas fórmulas):
// mismas operaciones IEEE 754 en ambos lados => mismos bits, por construcción.
// Dominio restringido (suficiente para Sinkhorn): x ∈ [1e-3, 1e3] y clamps
// de log-space [-0.3, 10.0]; exp argumentado en [exp(-0.3), exp(10)].

/// ln(x) determinista para x ∈ [1e-3, 1e3]. Reducción: x = m·2^k con m
/// normalizado a m ∈ [√½, √2) (branch única), luego ln(m) = 2·atanh(t) con
/// t = (m−1)/(m+1) ∈ [−0.1716, 0.1716], evaluado por su serie explícita.
/// Misma secuencia de ops en CUDA (kvarn_desc.cuh::kvarn_log_f64).
pub fn kvarnLog(x: f64) f64 {
    std.debug.assert(x > 0);
    const LN2: f64 = 0.693147180559945309417232121458176568;
    const SQRT_HALF_HI: f64 = 0.70710678118654752440; // redondeado hacia arriba

    // Reducción: normalizar m a [√½, 1) — branch única.
    const fr = std.math.frexp(x);
    var m = fr.significand; // m ∈ [0.5, 1)
    var k: i32 = fr.exponent;
    if (m < SQRT_HALF_HI) {
        m *= 2.0;
        k -= 1;
    }
    // m ∈ [√½, 1); |t| ≤ (√2−1)/(√2+1) ≈ 0.1716. Serie atanh por Horner:
    // atanh(t) = t·(1 + t²/3 + t⁴/5 + ... + t²⁴/25); el resto tras t²⁵
    // es < t²⁷/27 ≈ 1.7e-21 absoluto — irrelevante a f64.
    const t = (m - 1.0) / (m + 1.0);
    const t2 = t * t;
    const atanh = t * (1.0 + t2 * (1.0 / 3.0 + t2 * (1.0 / 5.0 + t2 * (1.0 / 7.0 + t2 * (1.0 / 9.0 + t2 * (1.0 / 11.0 + t2 * (1.0 / 13.0 + t2 * (1.0 / 15.0 + t2 * (1.0 / 17.0 + t2 * (1.0 / 19.0 + t2 * (1.0 / 21.0 + t2 * (1.0 / 23.0 + t2 * (1.0 / 25.0)))))))))))));
    const kf: f64 = @floatFromInt(k);
    return 2.0 * atanh + kf * LN2;
}

/// exp(x) determinista. Reducción: x = k·ln2 + r con k = round(x/ln2),
/// r ∈ [−ln2/2, ln2/2] POR CONSTRUCCIÓN (cualquier x finito); exp(r) por
/// serie de Taylor f64 (13 términos — error sub-ulp en |r| ≤ 0.3466),
/// escala final por 2^k. Válido para todo x finito cuyo resultado no
/// desborde f64 (el caller de Sinkhorn opera en log-space [-0.3, 10]).
pub fn kvarnExp(x: f64) f64 {
    std.debug.assert(std.math.isFinite(x));
    const LN2: f64 = 0.693147180559945309417232121458176568;
    const kf = x / LN2;
    const k: i32 = @intFromFloat(@round(kf));
    const kr: f64 = @floatFromInt(k);
    const r = x - kr * LN2; // |r| ≤ ln2/2 ≈ 0.3466
    // Taylor exp(r) en f64, 11 términos: término siguiente r¹¹/11! ≈ 8e-13
    // absoluto para |r| ≤ 0.347; relativo al resultado (≥ 0.7) < 1.2e-12…
    // se requiere < 1e-14 => 12 términos (r¹²/12! ≈ 2.3e-14 abs, rel 3e-14
    // en el peor caso) + el empuje final: evaluamos con más margen usando
    // 13 términos (r¹³/13! ≈ 6e-16, sub-ulp).
    const r2 = r * r;
    const r3 = r2 * r;
    const r4 = r2 * r2;
    const r5 = r3 * r2;
    const r6 = r3 * r3;
    const r7 = r4 * r3;
    const r8 = r4 * r4;
    const r9 = r5 * r4;
    const r10 = r5 * r5;
    const r11 = r6 * r5;
    const r12 = r6 * r6;
    const r13 = r7 * r6;
    const poly = 1.0 + r + r2 * (1.0 / 2.0) + r3 * (1.0 / 6.0) + r4 * (1.0 / 24.0) + r5 * (1.0 / 120.0) + r6 * (1.0 / 720.0) + r7 * (1.0 / 5040.0) + r8 * (1.0 / 40320.0) + r9 * (1.0 / 362880.0) + r10 * (1.0 / 3628800.0) + r11 * (1.0 / 39916800.0) + r12 * (1.0 / 479001600.0) + r13 * (1.0 / 6227020800.0);
    return std.math.ldexp(poly, k);
}

test "kvarn determinista: log/exp redondos al ulp en dominio Sinkhorn" {
    // El polinomio debe ser preciso a <1 ulp f64 frente al libm en el
    // dominio operativo real (std clamp [1e-3,1e3] y log-space [-0.3,10]).
    const xs = [_]f64{ 1e-3, 0.01, 0.1, 0.5, 0.7071067811865476, 1.0, 1.4142135623730951, 2.0, 3.7, 10.0, 100.0, 1e3 };
    for (xs) |x| {
        const got = kvarnLog(x);
        const want = @log(x);
        // x==1 => want==0: comparación absoluta en ulps; el resto relativa.
        if (want == 0.0) {
            try std.testing.expect(@abs(got) < 1e-300);
        } else {
            const rel = @abs(got - want) / @abs(want);
            try std.testing.expect(rel < 1e-14);
        }
    }
    const ls = [_]f64{ -0.3, -0.1, 0.0, 0.05, 0.5, 1.0, 2.5, 5.0, 10.0 };
    for (ls) |l| {
        const got = kvarnExp(l);
        const want = @exp(l);
        const rel = @abs(got - want) / @abs(want);
        try std.testing.expect(rel < 1e-14);
    }
}

// ============================================================================
// Codificación / Decodificación de tiles K y V (CPU reference)
// ============================================================================

/// Cuantiza un tile K (`group × head_dim` fp32) al payload K del record.
///
/// Tras la rotación Hadamard (responsabilidad del llamante) y la
/// normalización Sinkhorn, cuantiza por filas a `bits` (2..8) con
/// `q = round((balanced[r,c] - lo) / scale)`. Almacena:
///   - payload packed-bits en `record[layout.k_payload_off..]`
///   - `s_col_fp16[c]` en `record[layout.k_s_col_off..]`
///   - `zp_fp16[r] = s_row * lo` en `record[layout.k_zp_off..]`
///   - `s_row_fp16[r] = s_row * scale` en `record[layout.k_s_row_off..]`
///
/// Reusa el algoritmo transcrito de `llama_kvarn_quantize_tile` en C++.
/// Caller debe pre-rotar con `hadamard128Rows` (si head_dim=128) y copiar
/// `tile` en un buffer `balanced` separado si no quiere pisar el original.
pub fn encodeKTile(
    tile: []const f32,
    sinkhorn_iters: u32,
    bits: u8,
    layout: KvarnRecordLayout,
    record: []u8,
) !void {
    if (!isValidBits(bits)) return error.UnsupportedKvarBits;
    if (bits != layout.key_bits) return error.BitsMismatch;
    if (tile.len != KVAR_N_GROUP * layout.head_dim) return error.TileSizeMismatch;
    if (record.len < layout.k_payload_off + layout.k_payload_bytes) return error.RecordTooSmall;

    var balanced: [128 * 128]f32 = undefined;
    var s_col: [128]f32 = undefined;
    var s_row: [128]f32 = undefined;
    varianceNormalize(tile, sinkhorn_iters, &balanced, &s_col, &s_row);

    const qmax: f32 = @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1);
    var r: usize = 0;
    while (r < 128) : (r += 1) {
        const begin = r * 128;
        const end = begin + 128;
        var lo: f32 = balanced[begin];
        var hi: f32 = balanced[begin];
        var j: usize = begin + 1;
        while (j < end) : (j += 1) {
            const v = balanced[j];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        const range = hi - lo;
        const scale = if (range > 0) range / qmax else 1e-10;

        var c: usize = 0;
        while (c < 128) : (c += 1) {
            const value_f = @round((balanced[begin + c] - lo) / scale);
            const clamped = clamp(value_f, 0.0, qmax);
            const q: u8 = @intFromFloat(clamped);
            packBit(record[layout.k_payload_off..], r * 128 + c, bits, q);
        }

        const absorb = s_row[r];
        storeF16(record, layout.k_s_col_off, r, absorb * scale);
        storeF16(record, layout.k_zp_off, r, absorb * lo);
    }
    var c2: usize = 0;
    while (c2 < 128) : (c2 += 1) {
        storeF16(record, layout.k_s_row_off, c2, s_col[c2]);
    }
}

/// Decuantiza el payload K de un record al tile fp32 (`group × head_dim`).
///
/// `tile[r,c] = (q * s_row_scale + s_row_zp) * s_col` (transcripción directa
/// del dequantize K del upstream).
pub fn decodeKTile(
    record: []const u8,
    bits: u8,
    layout: KvarnRecordLayout,
    tile: []f32,
) !void {
    if (!isValidBits(bits)) return error.UnsupportedKvarBits;
    if (bits != layout.key_bits) return error.BitsMismatch;
    if (tile.len != KVAR_N_GROUP * layout.head_dim) return error.TileSizeMismatch;
    if (record.len < layout.k_payload_off + layout.k_payload_bytes) return error.RecordTooSmall;

    var r: usize = 0;
    while (r < 128) : (r += 1) {
        const scale = loadF16(record, layout.k_s_col_off, r);
        const zp = loadF16(record, layout.k_zp_off, r);
        var c: usize = 0;
        while (c < 128) : (c += 1) {
            const other = loadF16(record, layout.k_s_row_off, c);
            const q = unpackBit(record[layout.k_payload_off..], r * 128 + c, bits);
            tile[r * 128 + c] = (@as(f32, @floatFromInt(q)) * scale + zp) * other;
        }
    }
}

/// Cuantiza un tile V (`group × head_dim` fp32) al payload V del record.
/// Estructura equivalente a `encodeKTile` pero con `s_col` por columna (no
/// fila) y `s_row` absorbido en scale/zp. Ver transcripción C++ líneas 728-766.
pub fn encodeVTile(
    tile: []const f32,
    sinkhorn_iters: u32,
    bits: u8,
    layout: KvarnRecordLayout,
    record: []u8,
) !void {
    if (!isValidBits(bits)) return error.UnsupportedKvarBits;
    if (bits != layout.value_bits) return error.BitsMismatch;
    if (tile.len != KVAR_N_GROUP * layout.head_dim) return error.TileSizeMismatch;
    if (record.len < layout.v_payload_off + layout.v_payload_bytes) return error.RecordTooSmall;

    var balanced: [128 * 128]f32 = undefined;
    var s_col: [128]f32 = undefined;
    var s_row: [128]f32 = undefined;
    varianceNormalize(tile, sinkhorn_iters, &balanced, &s_col, &s_row);

    const qmax: f32 = @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1);
    var r: usize = 0;
    while (r < 128) : (r += 1) {
        const begin = r * 128;
        const end = begin + 128;
        var lo: f32 = balanced[begin];
        var hi: f32 = balanced[begin];
        var j: usize = begin + 1;
        while (j < end) : (j += 1) {
            const v = balanced[j];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        const range = hi - lo;
        const scale = if (range > 0) range / qmax else 1e-10;

        var c: usize = 0;
        while (c < 128) : (c += 1) {
            const value_f = @round((balanced[begin + c] - lo) / scale);
            const clamped = clamp(value_f, 0.0, qmax);
            const q: u8 = @intFromFloat(clamped);
            packBit(record[layout.v_payload_off..], r * 128 + c, bits, q);
        }

        storeF16(record, layout.v_s_row_off, r, s_row[r] * scale);
        storeF16(record, layout.v_zp_off, r, s_row[r] * lo);
    }
    var c2: usize = 0;
    while (c2 < 128) : (c2 += 1) {
        storeF16(record, layout.v_s_col_off, c2, s_col[c2]);
    }
}

/// Decuantiza el payload V de un record al tile fp32.
pub fn decodeVTile(
    record: []const u8,
    bits: u8,
    layout: KvarnRecordLayout,
    tile: []f32,
) !void {
    if (!isValidBits(bits)) return error.UnsupportedKvarBits;
    if (bits != layout.value_bits) return error.BitsMismatch;
    if (tile.len != KVAR_N_GROUP * layout.head_dim) return error.TileSizeMismatch;
    if (record.len < layout.v_payload_off + layout.v_payload_bytes) return error.RecordTooSmall;

    var r: usize = 0;
    while (r < 128) : (r += 1) {
        const scale = loadF16(record, layout.v_s_row_off, r);
        const zp = loadF16(record, layout.v_zp_off, r);
        var c: usize = 0;
        while (c < 128) : (c += 1) {
            const other = loadF16(record, layout.v_s_col_off, c);
            const q = unpackBit(record[layout.v_payload_off..], r * 128 + c, bits);
            tile[r * 128 + c] = (@as(f32, @floatFromInt(q)) * scale + zp) * other;
        }
    }
}

// ============================================================================
// 9.4 (lane-b) D2: acceso por slices (cabezas físicas) para head_dim > 128
// ============================================================================
// Diseño canónico beellama (kvarn.cu:1086-1105): una cabeza lógica D≥256
// se despliega en `slices = hd/128` cabezas FÍSICAS de 128. El store aplica
// WHT-128 intra-slice + cross-slice butterfly al vector del token y escribe
// cada slice como head física propia del stage/record; el cuantizado es
// SIEMPRE per (cabeza física, grupo) con layout 128 — exactamente
// `encodeKTile`/`encodeVTile` existentes. El kernel GPU portable lee
// records con `k_desc.head_base + slice` (portable.cu:874/938) ⇒ el manager
// que integra 9.4 debe dimensionar records como
// [layer][n_heads·slices][group] con layout hd=128 y extraer el slice s
// del tile lógico con estos helpers.
//
// El slice de un tile [128][hd] es el sub-tile [128][128]:
//   slice s: tile[r][s·128 + c] → slice_buf[r][c]
// (K es pre-rotado por `hadamardSlicesRows` completo; V tal cual — igual
// que el camino hd=128 con `hadamard128Rows`).

/// Extrae el slice `s` (sub-tile 128×128) de un tile lógico [128][hd].
pub fn extractSlice(tile: []const f32, head_dim: usize, slice: usize, out: *[128 * 128]f32) void {
    std.debug.assert(tile.len == KVAR_N_GROUP * head_dim);
    std.debug.assert(slice < head_dim / 128);
    var r: usize = 0;
    while (r < 128) : (r += 1) {
        var c: usize = 0;
        while (c < 128) : (c += 1) {
            out[r * 128 + c] = tile[r * head_dim + slice * 128 + c];
        }
    }
}

/// Inyecta el slice `s` (sub-tile 128×128) en un tile lógico [128][hd].
pub fn injectSlice(tile: []f32, head_dim: usize, slice: usize, src: *const [128 * 128]f32) void {
    std.debug.assert(tile.len == KVAR_N_GROUP * head_dim);
    std.debug.assert(slice < head_dim / 128);
    var r: usize = 0;
    while (r < 128) : (r += 1) {
        var c: usize = 0;
        while (c < 128) : (c += 1) {
            tile[r * head_dim + slice * 128 + c] = src[r * 128 + c];
        }
    }
}

// ============================================================================
// 9.12 (lane-b) F2: D64 rect — encode/decode CPU de tiles 64×128
// ============================================================================
// Espejo del diseño beellama d64 @e1f6d6fe6 (kvarn_d64_quantize_stage
// :1803): el tile de seal es RECTANGULAR con orientación según el lado:
//   K (value=0): rows=64 (dims), cols=128 (tokens) — dim-major, igual que
//               el seal d128 GPU (kvarn_seal_k_side transpose=1).
//   V (value=1): rows=128 (tokens), cols=64 (dims) — token-major (v_side
//               transpose=0).
// Sinkhorn rect: escalas s_col[cols], s_row[rows]; cuantización por FILA r
// (q index r*cols+c, LSB-first); axes del record C1 F1 (hd=64):
//   fila r: scale_axis[r]=s_row[r]·scale, zp_axis[r]=s_row[r]·lo
//   other_axis[c]=s_col[c]
// que mapea a K(k_s_col=r fila dim, k_zp=r, k_s_row=c token) y
// V(v_s_row=r token, v_zp=r, v_s_col=c dim) — mismo contrato D128.

/// Dimensiones del tile rect de seal D64 para un lado.
pub fn d64TileShape(side_k: bool) struct { rows: usize, cols: usize } {
    return if (side_k) .{ .rows = 64, .cols = 128 } else .{ .rows = 128, .cols = 64 };
}

/// Transpone un staging [128 tokens][64 dims] al tile rect del lado.
/// K: out[d*128 + t] = in[t*64 + d] (dim-major); V: out[t*64 + d] = in[t*64 + d].
pub fn d64StageToRect(staging: []const f32, side_k: bool, out: []f32) void {
    if (side_k) {
        std.debug.assert(staging.len == 128 * 64 and out.len == 64 * 128);
        var t: usize = 0;
        while (t < 128) : (t += 1) {
            var d: usize = 0;
            while (d < 64) : (d += 1) out[d * 128 + t] = staging[t * 64 + d];
        }
    } else {
        std.debug.assert(staging.len == 128 * 64 and out.len == 128 * 64);
        @memcpy(out, staging);
    }
}

/// Inverso de `d64StageToRect`: tile rect → staging [128][64].
pub fn d64RectToStage(rect: []const f32, side_k: bool, out: []f32) void {
    if (side_k) {
        std.debug.assert(rect.len == 64 * 128 and out.len == 128 * 64);
        var t: usize = 0;
        while (t < 128) : (t += 1) {
            var d: usize = 0;
            while (d < 64) : (d += 1) out[t * 64 + d] = rect[d * 128 + t];
        }
    } else {
        std.debug.assert(rect.len == 128 * 64 and out.len == 128 * 64);
        @memcpy(out, rect);
    }
}

/// Sinkhorn rect (rows×cols) — espejo de varianceNormalize con ejes
/// generalizados; misma aritmética (kvarnLog/kvarnExp f64, clamps, tie <=).
pub fn varianceNormalizeRect(
    tile: []const f32,
    rows: usize,
    cols: usize,
    sinkhorn_iters: u32,
    balanced: []f32,
    s_col: []f32,
    s_row: []f32,
) void {
    std.debug.assert(tile.len == rows * cols);
    std.debug.assert(balanced.len == rows * cols);
    std.debug.assert(s_col.len == cols and s_row.len == rows);

    var log_s_col_buf: [128]f32 = [_]f32{0.0} ** 128;
    var log_s_row_buf: [128]f32 = [_]f32{0.0} ** 128;
    const log_s_col = log_s_col_buf[0..cols];
    const log_s_row = log_s_row_buf[0..rows];

    @memset(s_col, 1.0);
    @memset(s_row, 1.0);
    var imbalance_best = imbalanceRect(tile, rows, cols);
    @memcpy(balanced, tile);

    var iter: u32 = 0;
    while (iter < sinkhorn_iters) : (iter += 1) {
        // Column pass (eje cols).
        var c: usize = 0;
        while (c < cols) : (c += 1) {
            var std_col = sampleStdRect(balanced, rows, cols, .col, c);
            if (std_col < 1e-3) std_col = 1e-3;
            if (std_col > 1e3) std_col = 1e3;
            const new_log = @as(f64, log_s_col[c]) + kvarnLog(@floatCast(std_col));
            log_s_col[c] = @floatCast(clampF64(new_log, -0.3, 10.0));
        }
        rebuildCurRect(tile, rows, cols, log_s_col, log_s_row, balanced);

        // Row pass (eje rows).
        var r: usize = 0;
        while (r < rows) : (r += 1) {
            var std_row = sampleStdRect(balanced, rows, cols, .row, r);
            if (std_row < 1e-3) std_row = 1e-3;
            if (std_row > 1e3) std_row = 1e3;
            const new_log = @as(f64, log_s_row[r]) + kvarnLog(@floatCast(std_row));
            log_s_row[r] = @floatCast(clampF64(new_log, -0.3, 10.0));
        }
        rebuildCurRect(tile, rows, cols, log_s_col, log_s_row, balanced);

        const imb = imbalanceRect(balanced, rows, cols);
        if (imb <= imbalance_best) {
            imbalance_best = imb;
            for (s_col, 0..) |*sc, i| sc.* = @floatCast(kvarnExp(log_s_col[i]));
            for (s_row, 0..) |*sr, i| sr.* = @floatCast(kvarnExp(log_s_row[i]));
        }
    }

    var r2: usize = 0;
    while (r2 < rows) : (r2 += 1) {
        var c2: usize = 0;
        while (c2 < cols) : (c2 += 1) {
            balanced[r2 * cols + c2] = tile[r2 * cols + c2] / (s_col[c2] * s_row[r2]);
        }
    }
}

const RectAxis = enum { row, col };

fn sampleStdRect(tile: []const f32, rows: usize, cols: usize, axis: RectAxis, i: usize) f32 {
    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    var n: usize = 0;
    switch (axis) {
        .row => {
            n = cols;
            var j: usize = 0;
            while (j < cols) : (j += 1) {
                const v = tile[i * cols + j];
                sum += v;
                sum_sq += @as(f64, v) * v;
            }
        },
        .col => {
            var r: usize = 0;
            while (r < rows) : (r += 1) {
                const v = tile[r * cols + i];
                sum += v;
                sum_sq += @as(f64, v) * v;
            }
            n = rows;
        },
    }
    const nf: f64 = @floatFromInt(n);
    const mean = sum / nf;
    const variance = if (n > 1) @max(0.0, (sum_sq - nf * mean * mean) / (nf - 1.0)) else 0.0;
    return @floatCast(@sqrt(variance));
}

fn rebuildCurRect(
    tile: []const f32,
    rows: usize,
    cols: usize,
    log_s_col: []const f32,
    log_s_row: []const f32,
    out: []f32,
) void {
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const sr = kvarnExp(log_s_row[r]);
        var c: usize = 0;
        while (c < cols) : (c += 1) {
            out[r * cols + c] = @floatCast(@as(f64, tile[r * cols + c]) / (kvarnExp(log_s_col[c]) * sr));
        }
    }
}

fn imbalanceRect(tile: []const f32, rows: usize, cols: usize) f32 {
    var col_min: f32 = std.math.inf(f32);
    var col_max: f32 = 0.0;
    var row_min: f32 = std.math.inf(f32);
    var row_max: f32 = 0.0;

    var c: usize = 0;
    while (c < cols) : (c += 1) {
        const s = sampleStdRect(tile, rows, cols, .col, c);
        if (s < col_min) col_min = s;
        if (s > col_max) col_max = s;
    }
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const s = sampleStdRect(tile, rows, cols, .row, r);
        if (s < row_min) row_min = s;
        if (s > row_max) row_max = s;
    }
    const col_min_safe = if (col_min < 1e-8) 1e-8 else col_min;
    const row_min_safe = if (row_min < 1e-8) 1e-8 else row_min;
    return col_max / col_min_safe + row_max / row_min_safe;
}

/// Encode K tile D64: `rect` dim-major [64 dims][128 tokens] (usar
/// `d64StageToRect(staging, true, ...)`). Pre-condición: K ya rotado con
/// WHT-64 por token (`hadamardSlicesRows` hd=64 sobre el staging).
pub fn encodeKTile64(
    rect: []const f32,
    sinkhorn_iters: u32,
    bits: u8,
    layout: KvarnRecordLayout,
    record: []u8,
) !void {
    try encodeRectSide(rect, 64, 128, sinkhorn_iters, bits, layout, record, .k);
}

/// Encode V tile D64: `rect` token-major [128 tokens][64 dims].
pub fn encodeVTile64(
    rect: []const f32,
    sinkhorn_iters: u32,
    bits: u8,
    layout: KvarnRecordLayout,
    record: []u8,
) !void {
    try encodeRectSide(rect, 128, 64, sinkhorn_iters, bits, layout, record, .v);
}

const RectSide = enum { k, v };

fn encodeRectSide(
    rect: []const f32,
    rows: usize,
    cols: usize,
    sinkhorn_iters: u32,
    bits: u8,
    layout: KvarnRecordLayout,
    record: []u8,
    side: RectSide,
) !void {
    if (!isValidBits(bits)) return error.UnsupportedKvarBits;
    const bits_ok = switch (side) {
        .k => bits == layout.key_bits and layout.head_dim == 64,
        .v => bits == layout.value_bits and layout.head_dim == 64,
    };
    if (!bits_ok) return error.BitsMismatch;
    if (rect.len != rows * cols) return error.TileSizeMismatch;
    const payload_off = switch (side) {
        .k => layout.k_payload_off,
        .v => layout.v_payload_off,
    };
    const payload_bytes = switch (side) {
        .k => layout.k_payload_bytes,
        .v => layout.v_payload_bytes,
    };
    if (record.len < payload_off + payload_bytes) return error.RecordTooSmall;

    var balanced: [128 * 128]f32 = undefined;
    var s_col_buf: [128]f32 = undefined;
    var s_row_buf: [128]f32 = undefined;
    const s_col = s_col_buf[0..cols];
    const s_row = s_row_buf[0..rows];
    varianceNormalizeRect(rect, rows, cols, sinkhorn_iters, balanced[0 .. rows * cols], s_col, s_row);

    const qmax: f32 = @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1);
    // Cuantización por fila r (contrato d64: fila del rect).
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const begin = r * cols;
        const end = begin + cols;
        var lo: f32 = balanced[begin];
        var hi: f32 = balanced[begin];
        var j: usize = begin + 1;
        while (j < end) : (j += 1) {
            const v = balanced[j];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        const range = hi - lo;
        const scale = if (range > 0) range / qmax else 1e-10;

        var c: usize = 0;
        while (c < cols) : (c += 1) {
            const value_f = @round((balanced[begin + c] - lo) / scale);
            const clamped = clamp(value_f, 0.0, qmax);
            const q: u8 = @intFromFloat(clamped);
            packBit(record[payload_off..], r * cols + c, bits, q);
        }

        // Axes C1 F1 (hd=64): fila r → scale/zp absorbidos; other = s_col.
        const absorb = s_row[r];
        switch (side) {
            // K: fila=dim → k_s_col[dim], k_zp[dim]; other k_s_row=token.
            .k => {
                storeF16(record, layout.k_s_col_off, r, absorb * scale);
                storeF16(record, layout.k_zp_off, r, absorb * lo);
            },
            // V: fila=token → v_s_row[token], v_zp[token]; other v_s_col=dim.
            .v => {
                storeF16(record, layout.v_s_row_off, r, absorb * scale);
                storeF16(record, layout.v_zp_off, r, absorb * lo);
            },
        }
    }
    var c2: usize = 0;
    while (c2 < cols) : (c2 += 1) {
        switch (side) {
            .k => storeF16(record, layout.k_s_row_off, c2, s_col[c2]),
            .v => storeF16(record, layout.v_s_col_off, c2, s_col[c2]),
        }
    }
}

/// Decode K record D64 → rect dim-major [64][128].
pub fn decodeKTile64(
    record: []const u8,
    bits: u8,
    layout: KvarnRecordLayout,
    rect: []f32,
) !void {
    try decodeRectSide(record, bits, layout, rect, .k);
}

/// Decode V record D64 → rect token-major [128][64].
pub fn decodeVTile64(
    record: []const u8,
    bits: u8,
    layout: KvarnRecordLayout,
    rect: []f32,
) !void {
    try decodeRectSide(record, bits, layout, rect, .v);
}

fn decodeRectSide(
    record: []const u8,
    bits: u8,
    layout: KvarnRecordLayout,
    rect: []f32,
    side: RectSide,
) !void {
    if (!isValidBits(bits)) return error.UnsupportedKvarBits;
    const bits_ok = switch (side) {
        .k => bits == layout.key_bits and layout.head_dim == 64,
        .v => bits == layout.value_bits and layout.head_dim == 64,
    };
    if (!bits_ok) return error.BitsMismatch;
    const rows: usize = if (side == .k) 64 else 128;
    const cols: usize = if (side == .k) 128 else 64;
    if (rect.len != rows * cols) return error.TileSizeMismatch;
    const payload_off = switch (side) {
        .k => layout.k_payload_off,
        .v => layout.v_payload_off,
    };
    const payload_bytes = switch (side) {
        .k => layout.k_payload_bytes,
        .v => layout.v_payload_bytes,
    };
    if (record.len < payload_off + payload_bytes) return error.RecordTooSmall;

    var r: usize = 0;
    while (r < rows) : (r += 1) {
        // Fila r: scale/zp absorbidos (fila del rect); other = eje cols.
        const scale = switch (side) {
            .k => loadF16(record, layout.k_s_col_off, r),
            .v => loadF16(record, layout.v_s_row_off, r),
        };
        const zp = switch (side) {
            .k => loadF16(record, layout.k_zp_off, r),
            .v => loadF16(record, layout.v_zp_off, r),
        };
        var c: usize = 0;
        while (c < cols) : (c += 1) {
            const other = switch (side) {
                .k => loadF16(record, layout.k_s_row_off, c),
                .v => loadF16(record, layout.v_s_col_off, c),
            };
            const q = unpackBit(record[payload_off..], r * cols + c, bits);
            rect[r * cols + c] = (@as(f32, @floatFromInt(q)) * scale + zp) * other;
        }
    }
}

// ============================================================================
// Bit packing (LSB-first, sin orden específico de bits)
// ============================================================================

/// Pack un valor de `bits` (≤8) en `dst` (LSB-first). `dst.len` debe ser ≥
/// `packedBytes(n_values, bits)` con `n_values > index`.
pub fn packBit(dst: []u8, index: usize, bits: u8, value: u8) void {
    std.debug.assert(bits > 0 and bits <= 8);
    // Para bits=8, la máscara natural es 0xFF (no shift). Para bits<8,
    // construimos la máscara con shifts seguros.
    const mask: u8 = if (bits == 8) 0xFF else (@as(u8, 1) << @intCast(bits)) - 1;
    const v = value & mask;
    const bit_offset = index * bits;
    var b: usize = 0;
    while (b < bits) : (b += 1) {
        const dst_bit = bit_offset + b;
        const bit_val: u8 = (v >> @intCast(b)) & 1;
        const pos_mask: u8 = @as(u8, 1) << @intCast(dst_bit % 8);
        if (bit_val != 0) {
            dst[dst_bit / 8] |= pos_mask;
        } else {
            dst[dst_bit / 8] &= ~pos_mask;
        }
    }
}

/// Unpack un valor de `bits` (≤8) en `src` (LSB-first).
pub fn unpackBit(src: []const u8, index: usize, bits: u8) u8 {
    std.debug.assert(bits > 0 and bits <= 8);
    var value: u8 = 0;
    const bit_offset = index * bits;
    var b: usize = 0;
    while (b < bits) : (b += 1) {
        const src_bit = bit_offset + b;
        const bit_val = (src[src_bit / 8] >> @intCast(src_bit % 8)) & 1;
        value |= @as(u8, @intCast(bit_val)) << @intCast(b);
    }
    return value;
}

// ============================================================================
// Almacenamiento fp16 dentro del record (layout canónico, LE)
// ============================================================================

fn storeF16(record: []u8, offset: usize, index: usize, value: f32) void {
    const fp16: f16 = @floatCast(value);
    const fp16_bits: u16 = @bitCast(fp16);
    const pos = offset + index * @sizeOf(u16);
    const bytes = std.mem.toBytes(fp16_bits);
    record[pos] = bytes[0];
    record[pos + 1] = bytes[1];
}

fn loadF16(record: []const u8, offset: usize, index: usize) f32 {
    const pos = offset + index * @sizeOf(u16);
    var bytes: [@sizeOf(u16)]u8 = undefined;
    bytes[0] = record[pos];
    bytes[1] = record[pos + 1];
    const bits: u16 = @bitCast(bytes);
    const fp16: f16 = @bitCast(bits);
    return @floatCast(fp16);
}

/// Conversión fp32 → fp16 (wrapper del fp16 nativo de Zig 0.16).
/// Mantenida por simetría con la API upstream; @floatCast produce el mismo
/// resultado bit-exacto que ggml_fp32_to_fp16 con RNE.
pub fn floatToF16(value: f32) u16 {
    const fp16: f16 = @floatCast(value);
    return @as(u16, @bitCast(fp16));
}

/// Conversión fp16 → fp32. Complemento de `floatToF16`.
pub fn f16ToFloat(h: u16) f32 {
    const fp16: f16 = @bitCast(h);
    return @floatCast(fp16);
}

// ============================================================================
// Tests
// ============================================================================

test "KvarnType parse roundtrip" {
    const t = KvarnType{ .key_bits = 5, .value_bits = 4 };
    const s = t.name();
    try std.testing.expectEqualStrings("kvarn_k5v4_g128", s);
    const back = KvarnType.parse(s).?;
    try std.testing.expectEqual(t.key_bits, back.key_bits);
    try std.testing.expectEqual(t.value_bits, back.value_bits);

    try std.testing.expect(KvarnType.parse("kvarn_k8v8_g128") != null);
    try std.testing.expect(KvarnType.parse("kvarn_k1v4_g128") == null); // bits no soportados
    try std.testing.expect(KvarnType.parse("foo") == null);
}

test "KvarnRecordLayout init alignment to 32B" {
    const layout = try KvarnRecordLayout.init(128, 4, 4);
    try std.testing.expect(layout.tile_bytes % KVAR_L2_SECTOR == 0);
    try std.testing.expect(layout.k_payload_off == 0);
    try std.testing.expect(layout.v_payload_off > layout.k_s_row_off);
    try std.testing.expect(layout.tile_bytes > layout.v_zp_off);
}

test "hadamard128InPlace isParseval" {
    // Hadamard normalizado preserva Parseval: ||x||² == ||Hx||².
    var values: [128]f32 = undefined;
    var i: usize = 0;
    while (i < 128) : (i += 1) values[i] = @as(f32, @floatFromInt(i)) * 0.01;
    var orig_norm: f64 = 0.0;
    for (values) |v| orig_norm += @as(f64, v) * @as(f64, v);

    hadamard128InPlace(&values);

    var new_norm: f64 = 0.0;
    for (values) |v| new_norm += @as(f64, v) * @as(f64, v);

    // Tolerancia 1e-3 por aritmética fp32.
    try std.testing.expectApproxEqAbs(orig_norm, new_norm, 1e-3);
}

test "hadamard128InPlace involutive" {
    // H·H = I: aplicar Hadamard dos veces recupera el original (signo + orden
    // por convención, pero H·H = I para Hadamard no normalizada; con normalización
    // 1/√128, aplicar dos veces da la identidad exacta).
    var values: [128]f32 = undefined;
    var original: [128]f32 = undefined;
    var i: usize = 0;
    while (i < 128) : (i += 1) {
        values[i] = @sin(@as(f32, @floatFromInt(i)) * 0.13);
        original[i] = values[i];
    }
    hadamard128InPlace(&values);
    hadamard128InPlace(&values);
    for (values, 0..) |v, idx| {
        try std.testing.expectApproxEqAbs(original[idx], v, 1e-3);
    }
}

test "floatToF16/f16ToFloat roundtrip" {
    const cases = [_]f32{ 0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 1e-3, 1e3, -1e3, 100.0, -100.0 };
    for (cases) |v| {
        const h = floatToF16(v);
        const back = f16ToFloat(h);
        const tol = @abs(v) * 1e-3 + 1e-3;
        try std.testing.expectApproxEqAbs(v, back, tol);
    }
}

test "packBit/unpackBit roundtrip" {
    const cases = [_]struct { bits: u8, vals: []const u8 }{
        .{ .bits = 2, .vals = &[_]u8{ 0, 1, 2, 3, 0, 3, 1, 2 } },
        .{ .bits = 4, .vals = &[_]u8{ 0, 15, 7, 8, 1, 14 } },
        .{ .bits = 5, .vals = &[_]u8{ 0, 31, 16, 15, 1 } },
        .{ .bits = 8, .vals = &[_]u8{ 0, 255, 128, 1, 127 } },
    };
    for (cases) |c| {
        const n = c.vals.len;
        const buf_size = packedBytes(n, c.bits);
        var buf = [_]u8{0} ** 256;
        for (c.vals, 0..) |v, i| packBit(&buf, i, c.bits, v);
        for (c.vals, 0..) |v, i| {
            const got = unpackBit(&buf, i, c.bits);
            try std.testing.expectEqual(v, got);
        }
        _ = buf_size;
    }
}

// ============================================================================
// 9.4 (lane-b) D2 ratification: cross-slice WHT por slices — CPU ref
// ============================================================================

test "hadamardSlicesRows D=256 isParseval" {
    // W = (I_S ⊗ H128)·C_S ortonormal ⇒ preserva la norma por fila.
    const hd = 256;
    var tile: [KVAR_N_GROUP * hd]f32 = undefined;
    var orig_norm: f64 = 0;
    var rng = std.Random.Xoshiro256.init(0xD2_0001);
    for (&tile) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    for (0..KVAR_N_GROUP) |r| {
        var row_norm: f64 = 0;
        for (0..hd) |c| {
            const v = tile[r * hd + c];
            row_norm += @as(f64, v) * @as(f64, v);
        }
        orig_norm += row_norm;
    }

    hadamardSlicesRows(&tile, hd);

    var new_norm: f64 = 0;
    for (0..KVAR_N_GROUP) |r| {
        var row_norm: f64 = 0;
        for (0..hd) |c| {
            const v = tile[r * hd + c];
            row_norm += @as(f64, v) * @as(f64, v);
        }
        new_norm += row_norm;
    }
    // Tolerancia relativa por fp32: 128 filas × 256 dims.
    try std.testing.expectApproxEqAbs(orig_norm, new_norm, orig_norm * 1e-4);
}

test "hadamardSlicesRows D=256 involutive" {
    // W·W = I: aplicar dos veces recupera el original.
    const hd = 256;
    var tile: [KVAR_N_GROUP * hd]f32 = undefined;
    var original: [KVAR_N_GROUP * hd]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(0xD2_0002);
    for (&tile) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    @memcpy(&original, &tile);

    hadamardSlicesRows(&tile, hd);
    hadamardSlicesRows(&tile, hd);

    var max_diff: f32 = 0;
    for (&tile, &original) |t, o| {
        max_diff = @max(max_diff, @abs(t - o));
    }
    // fp32: error de redondeo acumulado del butterfly (7+1 etapas).
    try std.testing.expect(max_diff < 1e-4);
}

test "hadamardSlicesRows D=256 equivale a wht_128-slices + butterfly manual" {
    // Oráculo de estructura: para una fila, el cross-slice butterfly de
    // SLICES=2 es (a+b, a-b)·1/√2 sobre pares (d, d+128) tras el WHT-128
    // de cada slice — espejo EXACTO de fattn_kvarn_wht_cross_slices<2>.
    const hd = 256;
    var tile: [KVAR_N_GROUP * hd]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(0xD2_0003);
    for (&tile) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    const orig_tile = tile;

    hadamardSlicesRows(&tile, hd);

    // Referencia manual: slice s de la fila r = H128(orig[r][s*128..]),
    // luego cross: out[0][d] = (a+b)/√2, out[1][d] = (a-b)/√2.
    var r: usize = 0;
    while (r < KVAR_N_GROUP) : (r += 1) {
        var s0: [128]f32 = undefined;
        var s1: [128]f32 = undefined;
        var d: usize = 0;
        while (d < 128) : (d += 1) {
            s0[d] = orig_tile[r * hd + d];
            s1[d] = orig_tile[r * hd + 128 + d];
        }
        hadamard128InPlace(&s0);
        hadamard128InPlace(&s1);
        d = 0;
        while (d < 128) : (d += 1) {
            const a = s0[d];
            const b = s1[d];
            const expect0 = (a + b) * 0.707106781186547524;
            const expect1 = (a - b) * 0.707106781186547524;
            try std.testing.expectApproxEqAbs(expect0, tile[r * hd + d], 1e-5);
            try std.testing.expectApproxEqAbs(expect1, tile[r * hd + 128 + d], 1e-5);
        }
    }
}

test "hadamardSlicesRows D=512 isParseval e involutiva" {
    // SLICES=4: butterfly doble (stride 1 y 2) + escala 1/2.
    const hd = 512;
    var tile: [KVAR_N_GROUP * hd]f32 = undefined;
    var original: [KVAR_N_GROUP * hd]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(0xD2_0004);
    for (&tile) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    @memcpy(&original, &tile);

    var orig_norm: f64 = 0;
    for (&original) |v| orig_norm += @as(f64, v) * @as(f64, v);
    hadamardSlicesRows(&tile, hd);
    var new_norm: f64 = 0;
    for (&tile) |v| new_norm += @as(f64, v) * @as(f64, v);
    try std.testing.expectApproxEqAbs(orig_norm, new_norm, orig_norm * 1e-4);

    hadamardSlicesRows(&tile, hd);
    var max_diff: f32 = 0;
    for (&tile, &original) |t, o| max_diff = @max(max_diff, @abs(t - o));
    try std.testing.expect(max_diff < 1e-4);
}

test "hadamardSlicesRows D=128 coincide con hadamard128Rows" {
    // SLICES=1: el butterfly cross-slice es no-op ⇒ equivalente al legacy.
    const hd = 128;
    var a: [KVAR_N_GROUP * hd]f32 = undefined;
    var b: [KVAR_N_GROUP * hd]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(0xD2_0005);
    for (&a) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    @memcpy(&b, &a);
    hadamard128Rows(&a, hd);
    hadamardSlicesRows(&b, hd);
    for (&a, &b) |x, y| try std.testing.expectEqual(x, y);
}

test "D2 roundtrip D=256: slices físicas + encode/decode per-slice" {
    // Cadena completa del diseño beellama: tile lógico [128][256] →
    // hadamardSlicesRows (K) → extract slice 0/1 → encodeKTile layout
    // hd=128 per cabeza física → decode → inject → hadamardSlicesRows
    // (involutiva) → tile ≈ original (gate SNR igual al manager lane-c:
    // ‖err‖/‖ref‖ ≤ 0.15 con bits=4).
    const hd: u32 = 256;
    const bits: u8 = 4;
    const layout = try KvarnRecordLayout.init(128, bits, bits);
    const rec_bytes = recordBytes(128, bits, bits);

    var tile: [KVAR_N_GROUP * 256]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(0xD2_0006);
    for (&tile) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    const original = tile;

    // K-side: rotación completa por slices.
    hadamardSlicesRows(&tile, hd);

    // Encode per cabeza física.
    var rec0: [512]u8 = [_]u8{0} ** 512;
    var rec1: [512]u8 = [_]u8{0} ** 512;
    var s0: [128 * 128]f32 = undefined;
    var s1: [128 * 128]f32 = undefined;
    extractSlice(&tile, 256, 0, &s0);
    extractSlice(&tile, 256, 1, &s1);
    try encodeKTile(&s0, 3, bits, layout, &rec0);
    try encodeKTile(&s1, 3, bits, layout, &rec1);
    try std.testing.expect(rec_bytes <= rec0.len);

    // Decode + inject.
    var d0: [128 * 128]f32 = undefined;
    var d1: [128 * 128]f32 = undefined;
    try decodeKTile(&rec0, bits, layout, &d0);
    try decodeKTile(&rec1, bits, layout, &d1);
    var recon: [KVAR_N_GROUP * 256]f32 = undefined;
    injectSlice(&recon, 256, 0, &d0);
    injectSlice(&recon, 256, 1, &d1);

    // Inversa de la rotación.
    hadamardSlicesRows(&recon, hd);

    // SNR relativo (gate del manager lane-c para kvarn4).
    var err: f64 = 0;
    var ref: f64 = 0;
    for (&recon, &original) |x, y| {
        err += @as(f64, x - y) * @as(f64, x - y);
        ref += @as(f64, y) * @as(f64, y);
    }
    const snr = @sqrt(err / ref);
    try std.testing.expect(snr < 0.15);
}
