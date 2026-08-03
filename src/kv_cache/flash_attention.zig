//! Integración de KV-cache cuantizado con FlashAttention
//! Gestiona de-cuantización on-the-fly y pasa tensores FP16 al kernel FA

const std = @import("std");
const qt = @import("quant_types.zig");
const gpu_dequant = @import("gpu_dequant.zig");

const QuantFormat = qt.QuantFormat;
const GpuDequantEngine = gpu_dequant.GpuDequantEngine;

/// Descriptor de operación FlashAttention
pub const FlashAttentionOp = struct {
    /// Batch size
    batch_size: u32,
    /// Número de cabezas de query
    num_q_heads: u32,
    /// Número de cabezas de key/value
    num_kv_heads: u32,
    /// Longitud de secuencia de query
    seq_len_q: u32,
    /// Longitud de secuencia de key/value (cache)
    seq_len_kv: u32,
    /// Dimensión por cabeza
    head_dim: u32,
    /// Escala softmax (1/sqrt(d))
    softmax_scale: f32,
    /// Máscara causal
    causal: bool,
    /// Dropout (0 = desactivado)
    dropout_p: f32,
};

/// Tensor en GPU para FlashAttention
pub const GPUTensor = struct {
    /// Puntero de device
    d_ptr: usize,
    /// Formato
    dtype: DType,
    /// Forma [batch, seq, heads, dim]
    shape: [4]u32,
    /// Strides en elementos
    strides: [4]u32,

    pub const DType = enum {
        fp16,
        fp32,
        bf16,
    };

    /// Número total de elementos
    pub fn numElements(self: GPUTensor) usize {
        var total: usize = 1;
        for (self.shape) |s| total *= s;
        return total;
    }

    /// Bytes por elemento
    pub fn bytesPerElement(self: GPUTensor) usize {
        return switch (self.dtype) {
            .fp16, .bf16 => 2,
            .fp32 => 4,
        };
    }

    /// Bytes totales
    pub fn totalBytes(self: GPUTensor) usize {
        return self.numElements() * self.bytesPerElement();
    }
};

/// Integrador KV-cache + FlashAttention
pub const KVFlashAttention = struct {
    /// Engine de de-cuantización GPU
    dequant_engine: ?GpuDequantEngine,
    /// Buffer de salida FA
    fa_output_buffer: ?GPUTensor,
    /// Buffer softmax LSE
    lse_buffer: ?[]f32,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .dequant_engine = null,
            .fa_output_buffer = null,
            .lse_buffer = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.lse_buffer) |buf| {
            self.allocator.free(buf);
        }
    }

    /// Configura el engine de de-cuantización
    pub fn setDequantEngine(self: *Self, engine: GpuDequantEngine) void {
        self.dequant_engine = engine;
    }

    /// Ejecuta FlashAttention con KV-cache cuantizado
    /// Los K/V de entrada están cuantizados; se de-cuantizan on-the-fly a FP16
    pub fn forward(
        self: *Self,
        op: FlashAttentionOp,
        /// Query en GPU [B, Sq, Hq, D] FP16
        d_q: GPUTensor,
        /// Key cuantizado en GPU
        d_k_quant: QuantizedGPUTensor,
        /// Value cuantizado en GPU
        d_v_quant: QuantizedGPUTensor,
        /// Output en GPU [B, Sq, Hq, D] FP16
        d_out: GPUTensor,
    ) !void {
        // 1. De-cuantizar K y V a FP16 en GPU
        const d_k_fp16 = try self.dequantizeKV(d_k_quant, op.seq_len_kv, op.head_dim);
        const d_v_fp16 = try self.dequantizeKV(d_v_quant, op.seq_len_kv, op.head_dim);

        // 2. Lanzar FlashAttention con K/V FP16
        try self.launchFlashAttention(op, d_q, d_k_fp16, d_v_fp16, d_out);

        // 3. Sincronizar
        // cudaStreamSynchronize(...)
    }

    /// De-cuantiza un tensor cuantizado en GPU a FP16
    fn dequantizeKV(
        self: *Self,
        tensor: QuantizedGPUTensor,
        seq_len: u32,
        head_dim: u32,
    ) !GPUTensor {
        const engine = self.dequant_engine orelse return error.NoGpuEngine;

        const num_elements = @as(usize, seq_len) * @as(usize, head_dim);
        const block_size = tensor.format.defaultBlockSize();

        const d_out = try engine.dequantize(
            tensor.format,
            tensor.d_raw,
            tensor.d_scales,
            tensor.d_zero_points,
            num_elements,
            block_size,
        );

        return GPUTensor{
            .d_ptr = d_out,
            .dtype = .fp16,
            .shape = .{ 1, seq_len, 1, head_dim },
            .strides = .{ seq_len * head_dim, head_dim, head_dim, 1 },
        };
    }

    /// Lanza el kernel FlashAttention (placeholder para integración real)
    fn launchFlashAttention(
        self: *Self,
        op: FlashAttentionOp,
        d_q: GPUTensor,
        d_k: GPUTensor,
        d_v: GPUTensor,
        d_out: GPUTensor,
    ) !void {
        _ = self;
        _ = op;
        _ = d_q;
        _ = d_k;
        _ = d_v;
        _ = d_out;
        // Placeholder: aquí se llamaría al kernel FlashAttention
        // Implementación real requiere bindings a cutlass/flash-attn
        std.log.debug("FlashAttention forward: Q={any} K={any} V={any} -> O={any}", .{
            d_q.shape, d_k.shape, d_v.shape, d_out.shape,
        });
    }

    /// Prepara buffers de salida para un tamaño máximo
    pub fn prepareOutputBuffer(self: *Self, max_batch: u32, max_seq: u32, max_heads: u32, head_dim: u32) !void {
        const num_elements = @as(usize, max_batch) * max_seq * max_heads * head_dim;
        const bytes = num_elements * 2; // FP16

        // Asumir asignación GPU via cudaz
        // const d_ptr = try cudaz.cuMemAlloc(bytes);
        _ = bytes;

        self.fa_output_buffer = GPUTensor{
            .d_ptr = 0, // d_ptr,
            .dtype = .fp16,
            .shape = .{ max_batch, max_seq, max_heads, head_dim },
            .strides = .{
                max_seq * max_heads * head_dim,
                max_heads * head_dim,
                head_dim,
                1,
            },
        };

        // LSE: [batch, heads, seq_q]
        const lse_elements = @as(usize, max_batch) * max_heads * max_seq;
        self.lse_buffer = try self.allocator.alloc(f32, lse_elements);
    }
};

/// Tensor cuantizado residente en GPU
pub const QuantizedGPUTensor = struct {
    format: QuantFormat,
    /// Puntero a datos cuantizados en device
    d_raw: usize,
    /// Puntero a scales en device (null si no aplica)
    d_scales: ?usize,
    /// Puntero a zero_points en device (null si no aplica)
    d_zero_points: ?usize,
    /// Número de elementos lógicos
    num_elements: usize,
    /// Tamaño de bloque
    block_size: usize,
};

/// Utilidades de shape para FlashAttention
pub const ShapeUtils = struct {
    /// Calcula strides row-major
    pub fn rowMajorStrides(shape: [4]u32) [4]u32 {
        var strides: [4]u32 = undefined;
        strides[3] = 1;
        strides[2] = shape[3];
        strides[1] = shape[2] * shape[3];
        strides[0] = shape[1] * shape[2] * shape[3];
        return strides;
    }

    /// Verifica compatibilidad de dimensiones para FA
    pub fn checkCompatible(q_shape: [4]u32, k_shape: [4]u32, v_shape: [4]u32) bool {
        // batch debe coincidir
        if (q_shape[0] != k_shape[0] or q_shape[0] != v_shape[0]) return false;
        // head_dim debe coincidir
        if (q_shape[3] != k_shape[3] or q_shape[3] != v_shape[3]) return false;
        // seq_len de K y V debe coincidir
        if (k_shape[1] != v_shape[1]) return false;
        return true;
    }
};
