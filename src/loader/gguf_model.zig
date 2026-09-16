//! GgufModel — envoltura de alto nivel sobre un archivo GGUF para inferencia.
//! Posee el archivo mmap, deriva la ModelConfig y ofrece carga de tensores
//! raíz (embedding, output_norm, lm_head) dequantizados a f16/f32.
const std = @import("std");
const gguf = @import("gguf");
const QuantWeight = @import("quant_weight").QuantWeight;
const model_config = @import("model_config");
const Tensor = @import("core").Tensor;
const kvcache = @import("kv_cache"); // lane-f F2A: decode/QuantFormat para LMQ40

pub const GgufModelError = error{
    MissingTensor,
    ModelTooLarge,
};

pub const GgufModel = struct {
    allocator: std.mem.Allocator,
    file: gguf.GgufFile,
    config: model_config.ModelConfig,

    const Self = @This();

    /// Carga el GGUF (mmap) y deriva la configuración del modelo.
    pub fn load(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Self {
        const dbg = @import("debug").dbg;
        dbg.printLevel(.info, "[loader] GgufModel.load: abriendo {s}\n", .{path});
        var file = try gguf.GgufFile.fromFileMmap(io, allocator, path);
        errdefer file.deinit();
        dbg.printLevel(.info, "[loader] mmap OK, tensors={d}\n", .{file.tensors.count()});
        const config = try model_config.ModelConfig.fromGguf(&file);
        dbg.printLevel(.info, "[loader] config OK arch={s}\n", .{config.architecture});
        return .{ .allocator = allocator, .file = file, .config = config };
    }

    /// 4.3' copy-once (lane-f coord): parsea el GGUF tomando prestado un
    /// buffer pinned YA cargado por el caller (`HostBank.fromFileCopyOnce`).
    /// Cero mmap, cero copia — RAM = 1× modelo. El caller mantiene el buffer
    /// (y su HostBank) vivo HASTA DESPUÉS del deinit de este GgufModel
    /// (orden: capas MoE → GgufModel → GgufFile borrowed → bank.unreg()).
    /// El engine MoE-aware (tickets F/C) usará esto + composeExternalScales-
    /// InPlace + moe_layer.g_pinned_preregistered antes de MoeLayer.init.
    pub fn loadFromRegion(allocator: std.mem.Allocator, region: []const u8) !Self {
        var file = try gguf.GgufFile.fromBytesBorrowed(allocator, region);
        errdefer file.deinit();
        const config = try model_config.ModelConfig.fromGguf(&file);
        return .{ .allocator = allocator, .file = file, .config = config };
    }

    pub fn deinit(self: *Self) void {
        self.file.deinit();
    }

    /// token_embd.weight -> Tensor(f16) [vocab, hidden] (row-major).
    /// Es una simple tabla de lookup por token, sin transposición.
    pub fn loadEmbedding(self: *const Self) !Tensor(f16) {
        const info = try self.findTensor("token_embd.weight", null);
        if (info.n_dims != 2) return GgufModelError.MissingTensor;
        const hidden: usize = @intCast(info.dims[0]);
        const vocab: usize = @intCast(info.dims[1]);
        return dequantToF16(self.allocator, info, self.file.tensorData(info), &.{ vocab, hidden });
    }

    /// token_embd.weight -> QuantWeight (bytes mmap sin dequantizar).
    /// Para el embedding cuant-residente 7.1d: embeddingLookupQuant
    /// dequantiza SOLO las filas de los tokens pedidos on-demand
    /// (cero materialización de la tabla f16 [vocab, hidden]).
    pub fn loadEmbeddingQuant(self: *const Self) !QuantWeight {
        const info = try self.findTensor("token_embd.weight", null);
        if (info.n_dims != 2) return GgufModelError.MissingTensor;
        return QuantWeight.init(info, self.file.tensorData(info));
    }

    /// output.weight -> Tensor(f16) [vocab, hidden].
    /// Usado como lm_head por linearProjection (trans_b=true), que espera [vocab, hidden].
    /// Si el modelo ata embeddings (sin output.weight), usa token_embd.weight.
    pub fn loadLmHead(self: *const Self) !Tensor(f16) {
        const info = self.findTensor("output.weight", null) catch
            (self.findTensor("token_embd.weight", null) catch return GgufModelError.MissingTensor);
        if (info.n_dims != 2) return GgufModelError.MissingTensor;
        const hidden: usize = @intCast(info.dims[0]);
        const vocab: usize = @intCast(info.dims[1]);
        return dequantToF16(self.allocator, info, self.file.tensorData(info), &.{ vocab, hidden });
    }

    /// output.weight -> QuantWeight Q4_0 (bytes mmap sin dequantizar, [in,out]).
    /// Para el GEMM cuantizado M=1 del decode. Si el modelo ata embeddings, usa
    /// token_embd.weight.
    pub fn loadLmHeadQuant(self: *const Self) !QuantWeight {
        const info = self.findTensor("output.weight", null) catch
            (self.findTensor("token_embd.weight", null) catch return GgufModelError.MissingTensor);
        if (info.n_dims != 2) return GgufModelError.MissingTensor;
        return QuantWeight.init(info, self.file.tensorData(info));
    }

    /// B6 — output.weight cuantizado ON-LOAD a q8_0 [vocab, hidden].
    /// Lee elems desde el mmap (BF16/F16/F32) fila a fila sin materializar
    /// el tensor completo: host RAM += solo el buffer cuantizado (~53%).
    /// Devuelve bytes [vocab * KB*34] con layout GGUF por fila:
    /// KB = hidden/32 bloques de [d f16][i8×32]. `hidden` debe ser %32.
    pub fn loadLmHeadQ80(self: *const Self, gpa: std.mem.Allocator) !struct {
        bytes: []u8,
        vocab: usize,
        hidden: usize,
        from_dtype: gguf.GgmlType,
    } {
        const info = self.findTensor("output.weight", null) catch
            (self.findTensor("token_embd.weight", null) catch return GgufModelError.MissingTensor);
        if (info.n_dims != 2) return GgufModelError.MissingTensor;
        const hidden: usize = @intCast(info.dims[0]);
        const vocab: usize = @intCast(info.dims[1]);
        if (hidden % 32 != 0) return GgufModelError.ModelTooLarge;
        const kb = hidden / 32;
        const row_out = kb * 34;

        const src_bytes = self.file.tensorData(info);
        // row stride en bytes del dtype fuente
        const elem_sz: usize = switch (info.dtype) {
            .f32 => 4,
            .f16, .bf16 => 2,
            else => return GgufModelError.ModelTooLarge, // ya cuantizado: usar loadLmHeadQuant
        };
        const row_in = hidden * elem_sz;

        const out = try gpa.alloc(u8, vocab * row_out);
        errdefer gpa.free(out);

        // Buffer de fila en heap: hidden puede ser 5120+ (27B) — el array
        // de stack [4096] desbordaba.
        const row_f32 = try gpa.alloc(f32, hidden);
        defer gpa.free(row_f32);

        for (0..vocab) |j| {
            const src_row = src_bytes[j * row_in ..][0..row_in];
            // dequant elems de la fila a f32
            switch (info.dtype) {
                .f32 => {
                    const f = std.mem.bytesAsSlice(f32, src_row);
                    @memcpy(row_f32[0..hidden], f);
                },
                .f16 => {
                    for (0..hidden) |c| {
                        const h = std.mem.readInt(u16, src_row[c * 2 ..][0..2], .little);
                        row_f32[c] = @floatCast(@as(f16, @bitCast(h)));
                    }
                },
                .bf16 => {
                    for (0..hidden) |c| {
                        const b = std.mem.readInt(u16, src_row[c * 2 ..][0..2], .little);
                        const bits: u32 = @as(u32, b) << 16;
                        row_f32[c] = @bitCast(bits);
                    }
                },
                else => unreachable,
            }
            // cuantizar por bloques de 32 → q8_0 canónico [d f16@0][i8×32@2]
            // (34B/bloque; el layout previo d f32@0/qs@4 = 36B desbordaba la
            // fila y era ilegible por el kernel B6 — fix lane-c, ver HANDOFFS)
            const dst_row = out[j * row_out ..][0..row_out];
            for (0..kb) |kb_i| {
                const b0 = kb_i * 32;
                var amax: f32 = 0;
                for (0..32) |c| amax = @max(amax, @abs(row_f32[b0 + c]));
                const d: f32 = if (amax > 0) amax / 127.0 else 1.0;
                const d16: u16 = @bitCast(@as(f16, @floatCast(d)));
                std.mem.writeInt(u16, dst_row[kb_i * 34 ..][0..2], d16, .little);
                for (0..32) |c| {
                    var q: i32 = @intFromFloat(@round(row_f32[b0 + c] / d));
                    q = @min(@max(q, -127), 127);
                    dst_row[kb_i * 34 + 2 + c] = @bitCast(@as(i8, @intCast(q)));
                }
            }
        }
        return .{ .bytes = out, .vocab = vocab, .hidden = hidden, .from_dtype = info.dtype };
    }

    /// LMQ40 (lane-f F2A): output.weight re-cuantizado ON-LOAD a q4_0
    /// [vocab, hidden] desde CUALQUIER dtype fuente (f16/bf16/f32/q4_0/q6_k/
    /// q8_0/...). Hermano del loadLmHeadQ80 (lane-c) pero destino q4_0:
    /// [vocab][KB*18] con layout GGUF por fila (KB = hidden/32, bloques
    /// [d f16][nibbles×32]). Consumidor: q4gemmLinear (+LMSPLIT mmq).
    /// La fila fuente q*_k se dequantiza por SB de 256 vía kv_quant.decode
    /// sin materializar el tensor f16 completo.
    pub fn loadLmHeadQ40(self: *const Self, gpa: std.mem.Allocator) !struct {
        bytes: []u8,
        vocab: usize,
        hidden: usize,
        from_dtype: gguf.GgmlType,
    } {
        const info = self.findTensor("output.weight", null) catch
            (self.findTensor("token_embd.weight", null) catch return GgufModelError.MissingTensor);
        if (info.n_dims != 2) return GgufModelError.MissingTensor;
        const hidden: usize = @intCast(info.dims[0]);
        const vocab: usize = @intCast(info.dims[1]);
        if (hidden % 32 != 0) return GgufModelError.ModelTooLarge;
        const kb = hidden / 32;
        const row_out = kb * 18;

        const src_bytes = self.file.tensorData(info);
        const src_fmt: kvcache.quant_types.QuantFormat = quantFormatOf(info.dtype) orelse
            return GgufModelError.ModelTooLarge;
        const src_bpb = src_fmt.bytesPerBlock(); // bytes por bloque fuente
        const src_bs = src_fmt.defaultBlockSize(); // valores por bloque fuente

        const out = try gpa.alloc(u8, vocab * row_out);
        errdefer gpa.free(out);

        const row_f16 = try gpa.alloc(f16, hidden);
        defer gpa.free(row_f16);

        const blocks_in = (hidden + src_bs - 1) / src_bs;
        const row_in = blocks_in * src_bpb;
        for (0..vocab) |j| {
            const src_row = src_bytes[j * row_in ..][0..row_in];
            // dequant fila completa → f16 (chunked dentro de decode)
            kvcache.kv_quant.decode(src_fmt, src_row, row_f16);
            // encode q4_0 canónico vía kv_quant.encode (SPLIT-16: elems
            // [0,16)=nibbles bajos, [16,32)=altos — NO intercalado por pares;
            // la 1ª versión intercalada producía logits basura). Misma
            // rutina testeada del KV-cache → paridad por construcción.
            const dst_row = out[j * row_out ..][0..row_out];
            kvcache.kv_quant.encode(.q4_0, row_f16[0..], dst_row[0 .. kb * 18]);
        }
        return .{ .bytes = out, .vocab = vocab, .hidden = hidden, .from_dtype = info.dtype };
    }

    fn quantFormatOf(dtype: gguf.GgmlType) ?kvcache.quant_types.QuantFormat {
        return switch (dtype) {
            .f16 => .fp16,
            // bf16 NO mapea a fp16 (layouts distintos) — LMQ40 lo rechaza;
            // bf16 usa el camino LMQ80 de lane-c.
            .f32 => .fp32,
            .q4_0 => .q4_0,
            .q4_1 => .q4_1,
            .q5_0 => .q5_0,
            .q5_1 => .q5_1,
            .q8_0 => .q8_0,
            .q6_k => .q6_k,
            .q4_k => .q4_k,
            .q5_k => .q5_k,
            .q3_k => .q3_k,
            .q2_k => .q2_k,
            else => null,
        };
    }

    /// output_norm.weight -> Tensor(f32) [hidden] (gamma de RMSNorm final).
    /// Si no existe, usa token_embd_norm.weight (naming alternativo).
    pub fn loadOutputNorm(self: *const Self) !Tensor(f32) {
        const info = self.findTensor("output_norm.weight", null) catch
            (self.findTensor("token_embd_norm.weight", null) catch return GgufModelError.MissingTensor);
        const numel: usize = @intCast(info.numel());
        const f32buf = try self.allocator.alloc(f32, numel);
        defer self.allocator.free(f32buf);
        try gguf.dequantTensor(info, self.file.tensorData(info), f32buf);
        const tensor = try Tensor(f32).initUninitialized(self.allocator, &.{numel});
        @memcpy(tensor.data, f32buf);
        return tensor;
    }

    /// Busca un tensor por nombre con un prefijo opcional.
    fn findTensor(self: *const Self, name: []const u8, prefix: ?[]const u8) !*const gguf.TensorInfo {
        const full = if (prefix) |p|
            try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ p, name })
        else
            name;
        defer if (prefix != null) self.allocator.free(full);
        return self.file.getTensor(full) orelse GgufModelError.MissingTensor;
    }

    /// C6-infra — dual-load del sidecar DFlash/DSpark/DFlash2 con herencia
    /// del target: el sidecar NO trae `tok_embd`/`output.weight`; las APIs
    /// de embedding/lm_head delegan al target (sin copias). Los encoders
    /// device-resident llegan en C6.x (requiere kernel no-causal de lane-a).
    pub fn loadSidecarDraft(target: *const Self, io: std.Io, allocator: std.mem.Allocator, sidecar_path: []const u8) !SidecarDraft {
        var m = try GgufModel.load(io, allocator, sidecar_path);
        errdefer m.deinit();

        // dflash.block_size (default 16)
        const block_size: usize = blk: {
            const v = m.file.getMeta("dflash.block_size") orelse break :blk 16;
            break :blk switch (v) {
                .uint32 => |x| @intCast(x),
                .int32 => |x| @intCast(x),
                .uint64 => |x| @intCast(x),
                .int64 => |x| @intCast(x),
                .string => |s| std.fmt.parseInt(usize, s, 10) catch 16,
                else => 16,
            };
        };

        // target_layers: array GGUF de índices de capa a tapear. 5.1
        // (lane-c): el conversor upstream escribe "dflash.target_layers"
        // (prefijo del arch, ver llama-arch LLM_KV_TARGET_LAYERS "%s.
        // target_layers") — el sidecar real qwen35-9b-dflash lo lleva ASÍ;
        // "attention.target_layers" es el nombre legacy de nuestros
        // primeros fixtures. Aceptar ambos. (5.2 lane-b1: mismo hallazgo.)
        var target_layers: []i32 = &[_]i32{};
        const tl_meta = m.file.getMeta("dflash.target_layers") orelse m.file.getMeta("attention.target_layers");
        if (tl_meta) |v| {
            if (v == .array) {
                const arr = v.array;
                var list = try allocator.alloc(i32, arr.items.len);
                var n: usize = 0;
                errdefer allocator.free(list);
                for (arr.items) |item| {
                    const x: i64 = switch (item) {
                        .uint8 => |z| @intCast(z),
                        .int8 => |z| @intCast(z),
                        .uint16 => |z| @intCast(z),
                        .int16 => |z| @intCast(z),
                        .uint32 => |z| @intCast(z),
                        .int32 => |z| @intCast(z),
                        .uint64 => |z| @intCast(z),
                        .int64 => |z| z,
                        else => continue,
                    };
                    list[n] = @intCast(x);
                    n += 1;
                }
                target_layers = list[0..n];
            }
        }

        // Conteo de capas del sidecar (blk.{0..N}) para el banner/validación.
        var n_layers: usize = 0;
        var it = m.file.tensors.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (!std.mem.startsWith(u8, name, "blk.")) continue;
            const dot = std.mem.indexOfScalarPos(u8, name, 4, '.') orelse continue;
            const idx = std.fmt.parseInt(usize, name[4..dot], 10) catch continue;
            if (idx + 1 > n_layers) n_layers = idx + 1;
        }

        return .{
            .model = m,
            .target = target,
            .block_size = block_size,
            .target_layers = target_layers,
            .n_layers = n_layers,
        };
    }
};

/// C6 — sidecar draft dual-load. Vida: el caller hace `deinit()` (libera el
/// mmap y los arrays propios; el target NO se toca).
pub const SidecarDraft = struct {
    model: GgufModel,
    /// Target del que se hereda tok_embd/lm_head/output_norm.
    target: *const GgufModel,
    /// dflash.block_size (default 16).
    block_size: usize,
    /// attention.target_layers (capas del target cuyas features alimenta
    /// el encoder). Vacío si el sidecar no lo trae.
    target_layers: []i32,
    /// Capas blk.* presentes en el sidecar.
    n_layers: usize,

    pub fn deinit(self: *SidecarDraft) void {
        if (self.target_layers.len > 0) self.model.allocator.free(self.target_layers);
        self.model.deinit();
    }

    /// tok_embd HEREDADO del target (el sidecar no lo trae).
    pub fn loadEmbedding(self: *const SidecarDraft) !Tensor(f16) {
        return self.target.loadEmbedding();
    }

    /// lm_head HEREDADO del target.
    pub fn loadLmHead(self: *const SidecarDraft) !Tensor(f16) {
        return self.target.loadLmHead();
    }

    /// output_norm HEREDADO del target (fallback propio si existiera).
    pub fn loadOutputNorm(self: *const SidecarDraft) !Tensor(f32) {
        return self.target.loadOutputNorm();
    }

    /// Encoder device-resident (fc[n_enc×n_embd]+RMSNorm): C6.x.
    pub fn loadEncoderWeights() !void {
        return error.NotImplemented;
    }

    /// KV-inject (wk/wv → cache draft durante prefill del target): C6.x.
    pub fn kvInject() !void {
        return error.NotImplemented;
    }
};

/// Dequantiza un tensor GGUF a f16 con el shape dado (row-major).
pub fn dequantToF16(
    allocator: std.mem.Allocator,
    info: *const gguf.TensorInfo,
    bytes: []const u8,
    shape: []const usize,
) !Tensor(f16) {
    var total: usize = 1;
    for (shape) |s| total *= s;
    if (total != info.numel()) return GgufModelError.ModelTooLarge;

    const f32buf = try allocator.alloc(f32, total);
    defer allocator.free(f32buf);
    try gguf.dequantTensor(info, bytes, f32buf);

    const tensor = try Tensor(f16).initUninitialized(allocator, shape);
    for (tensor.data, f32buf) |*d, s| d.* = @floatCast(s);
    return tensor;
}
