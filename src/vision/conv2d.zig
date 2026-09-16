//! Conv2D para patch embedding del ViT (Qwen2/3-VL, SigLIP).
//!
//! Layout del kernel en GGUF (llama.cpp ggml conv_2d, ops.cpp:6944-6947):
//!   kernel: [KW, KH, IC, OC] — ne[0]=KW contiguo, OC es la dim externa.
//!   En memoria lineal (row-major GGUF dims: dims[0] contigua):
//!     idx(oc, ic, ky, kx) = ((oc*IC + ic)*KH + ky)*KW + kx
//!   ⚠ dims GGUF: dims[0]=KW, dims[1]=KH, dims[2]=IC, dims[3]=OC.
//!
//! Layout de entrada/salida zig-ai (CHW, el estándar del preprocess):
//!   input:  [3, H, W] (CHW)
//!   output: [OC, OH, OW] (CHW)
//!
//! El patch embed del ViT usa stride=patch_size, sin padding, sin dilation,
//! kH=kW=patch_size, por lo que OH=H/patch, OW=W/patch.
//!
//! Estrategia MVP: convolución directa (naive). El patch-embed es UNA
//! convolución por imagen (no por capa), así que el costo es irrelevante
//! frente a N capas de transformer. La versión im2col+GEMM queda como
//! optimización futura (ver NOTA al final).
const std = @import("std");
const matmul = @import("matmul");
const Tensor = @import("core").Tensor;
const debugz = @import("debug");

pub const Conv2dError = error{
    ShapeMismatch,
    UnsupportedStride,
    OutOfMemory,
};

/// Conv2D directa (stride=k, sin pad). Peso en layout GGUF
/// [KW, KH, IC, OC] (dims[0]=KW contiguo — ver idx() arriba).
///
/// - `input`: [IC, H, W] CHW f32
/// - `weight_gguf`: bytes del tensor GGUF dequantizados a f32 (idx GGUF)
/// - `bias`: [OC] opcional (Qwen3-VL trae v.patch_embd.bias)
/// - `output`: [OC, OH, OW] con OH=(H-pad*2-kH)/stride+1... aquí sin pad:
///   OH=H/stride, OW=W/stride (H,W múltiplos de stride — lo garantiza
///   smartResize con múltiplos de patch·merge).
pub fn conv2dDirect(
    input: []const f32, // [IC, H, W]
    ic: usize,
    h: usize,
    w: usize,
    weight_gguf: []const f32, // [KW, KH, IC, OC] layout GGUF
    kw: usize,
    kh: usize,
    oc: usize,
    bias: ?[]const f32, // [OC]
    output: []f32, // [OC, OH, OW]
    stride: usize,
) Conv2dError!void {
    if (stride == 0) return Conv2dError.UnsupportedStride;
    const oh = h / stride;
    const ow = w / stride;
    if (input.len != ic * h * w) return Conv2dError.ShapeMismatch;
    if (weight_gguf.len != kw * kh * ic * oc) return Conv2dError.ShapeMismatch;
    if (output.len != oc * oh * ow) return Conv2dError.ShapeMismatch;

    // dst(o, oy, ox) = bias[o] + Σ_{ic,ky,kx} src(ic, oy·s+ky, ox·s+kx) ·
    //                  kernel(idx(o, ic, ky, kx))
    for (0..oc) |o| {
        const base: f32 = if (bias) |b| b[o] else 0.0;
        for (0..oh) |oy| {
            for (0..ow) |ox| {
                var acc: f32 = base;
                for (0..ic) |c| {
                    for (0..kh) |ky| {
                        for (0..kw) |kx| {
                            const sy = oy * stride + ky;
                            const sx = ox * stride + kx;
                            const sv = input[(c * h + sy) * w + sx];
                            // idx GGUF: ((o*IC + c)*KH + ky)*KW + kx
                            const wv = weight_gguf[o * (ic * kh * kw) + c * (kh * kw) + ky * kw + kx];
                            acc += sv * wv;
                        }
                    }
                }
                output[(o * oh + oy) * ow + ox] = acc;
            }
        }
    }
}

/// Conv2D con peso ya dequantizado a Tensor f32 en layout NATURAL
/// [OC, IC, KH, KW] (row-major trasponer del GGUF — lo que produce
/// dequantToF32 con shape {oc, ic*kh*kw} requiere cuidado: usar
/// conv2dDirect con los bytes lineales GGUF en su lugar).
///
/// Esta variante existe para tests y para pesos f32 almacenados natural.
pub fn conv2dNatural(
    input: []const f32, // [IC, H, W]
    ic: usize,
    h: usize,
    w: usize,
    weight_nat: []const f32, // [OC, IC, KH, KW]
    kw: usize,
    kh: usize,
    oc: usize,
    bias: ?[]const f32,
    output: []f32, // [OC, OH, OW]
    stride: usize,
) Conv2dError!void {
    if (stride == 0) return Conv2dError.UnsupportedStride;
    const oh = h / stride;
    const ow = w / stride;
    if (input.len != ic * h * w) return Conv2dError.ShapeMismatch;
    if (weight_nat.len != oc * ic * kh * kw) return Conv2dError.ShapeMismatch;
    if (output.len != oc * oh * ow) return Conv2dError.ShapeMismatch;

    for (0..oc) |o| {
        const base: f32 = if (bias) |b| b[o] else 0.0;
        for (0..oh) |oy| {
            for (0..ow) |ox| {
                var acc: f32 = base;
                for (0..ic) |c| {
                    for (0..kh) |ky| {
                        for (0..kw) |kx| {
                            const sv = input[(c * h + (oy * stride + ky)) * w + (ox * stride + kx)];
                            const wv = weight_nat[o * (ic * kh * kw) + c * (kh * kw) + ky * kw + kx];
                            acc += sv * wv;
                        }
                    }
                }
                output[(o * oh + oy) * ow + ox] = acc;
            }
        }
    }
}

/// Reordena un kernel Conv2D de layout GGUF [KW,KH,IC,OC] (dims[0] contiguo)
/// a layout natural [OC,IC,KH,KW]. Útil si se prefiere consumir natural.
pub fn kernelGgufToNatural(
    gguf_bytes: []const f32, // [KW*KH*IC*OC] layout GGUF
    out: []f32, // [OC*IC*KH*KW] natural
    kw: usize,
    kh: usize,
    ic: usize,
    oc: usize,
) void {
    for (0..oc) |o| {
        for (0..ic) |c| {
            for (0..kh) |ky| {
                for (0..kw) |kx| {
                    const src = o * (ic * kh * kw) + c * (kh * kw) + ky * kw + kx;
                    const dst = o * (ic * kh * kw) + c * (kh * kw) + ky * kw + kx;
                    out[dst] = gguf_bytes[src];
                }
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

const testing = std.testing;

test "conv2dDirect: kernel 2x2 stride 2, 1 canal → promedio por patch" {
    // Input 4x4, valores = índice
    const h = 4;
    const w = 4;
    const ic = 1;
    var input: [16]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @floatFromInt(i);

    // Kernel GGUF [KW=2, KH=2, IC=1, OC=1] — todos 0.25 (promedio)
    const kw = 2;
    const kh = 2;
    const oc = 1;
    const weight = [_]f32{0.25} ** (kw * kh * ic * oc);

    var output: [4]f32 = undefined; // [OC, 2, 2]
    try conv2dDirect(&input, ic, h, w, &weight, kw, kh, oc, null, &output, 2);

    // Patch (0,0): input 0,1,4,5 → promedio 2.5
    try testing.expectApproxEqAbs(2.5, output[0], 1e-6);
    // Patch (0,1): 2,3,6,7 → 4.5
    try testing.expectApproxEqAbs(4.5, output[1], 1e-6);
    // Patch (1,0): 8,9,12,13 → 10.5
    try testing.expectApproxEqAbs(10.5, output[2], 1e-6);
    // Patch (1,1): 10,11,14,15 → 12.5
    try testing.expectApproxEqAbs(12.5, output[3], 1e-6);
}

test "conv2dDirect: bias se suma" {
    const input = [_]f32{ 1, 2, 3, 4 }; // [1, 2, 2]
    const weight = [_]f32{1.0} ** 4; // kernel 2x2 identidad (suma)
    const bias = [_]f32{10.0};
    var output: [1]f32 = undefined;
    try conv2dDirect(&input, 1, 2, 2, &weight, 2, 2, 1, &bias, &output, 2);
    try testing.expectApproxEqAbs(10.0 + 1 + 2 + 3 + 4, output[0], 1e-6);
}

test "conv2dDirect: layout GGUF vs natural con kernel asimétrico" {
    // Kernel asimétrico para detectar transposición errónea:
    // solo kx=0,ky=0 pesa (1.0), resto 0. En GGUF [KW,KH,IC,OC]:
    // idx(o=0,c=0,ky=0,kx=0) = 0 → weight[0]=1.
    const h = 2;
    const w = 2;
    const input = [_]f32{ 10, 20, 30, 40 };

    var gguf_w = [_]f32{0} ** 4;
    gguf_w[0] = 1.0; // (o0,c0,ky0,kx0)
    var out_gguf: [1]f32 = undefined;
    try conv2dDirect(&input, 1, h, w, &gguf_w, 2, 2, 1, null, &out_gguf, 2);
    // Debe tomar input(0,0)=10
    try testing.expectApproxEqAbs(10.0, out_gguf[0], 1e-6);

    // Natural: weight_nat[(o·IC+c)·KH·KW + ky·KW + kx] → mismo índice lineal
    var nat_w = [_]f32{0} ** 4;
    nat_w[0] = 1.0;
    var out_nat: [1]f32 = undefined;
    try conv2dNatural(&input, 1, h, w, &nat_w, 2, 2, 1, null, &out_nat, 2);
    try testing.expectApproxEqAbs(10.0, out_nat[0], 1e-6);
}

test "conv2dDirect: multi-canal" {
    // IC=3 (RGB-like), kernel detecta canal 2 (index 2)
    const h = 2;
    const w = 2;
    const ic = 3;
    const input = [_]f32{ 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6 }; // [3, 2, 2]
    // Canal 0: [1,1,2,2], canal 1: [3,3,4,4], canal 2: [5,5,6,6]

    // Kernel GGUF [KW=2, KH=2, IC=3, OC=1]: solo canal 2 suma
    var gguf_w = [_]f32{0} ** (2 * 2 * 3 * 1);
    for (0..4) |i| gguf_w[2 * 2 * 2 + i] = 1.0; // canal 2
    var out: [1]f32 = undefined;
    try conv2dDirect(&input, ic, h, w, &gguf_w, 2, 2, 1, null, &out, 2);
    try testing.expectApproxEqAbs(5 + 5 + 6 + 6, out[0], 1e-6);
}
