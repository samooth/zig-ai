//! GgufModel — envoltura de alto nivel sobre un archivo GGUF para inferencia.
//! Posee el archivo mmap, deriva la ModelConfig y ofrece carga de tensores
//! raíz (embedding, output_norm, lm_head) dequantizados a f16/f32.
const std = @import("std");
const gguf = @import("gguf");
const QuantWeight = @import("quant_weight").QuantWeight;
const model_config = @import("model_config");
const Tensor = @import("core").Tensor;

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
        var file = try gguf.GgufFile.fromFileMmap(io, allocator, path);
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
