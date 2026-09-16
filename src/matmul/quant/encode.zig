//! Zig wrapper for the Phase 3 quantization (encode) kernels
//! (`src/matmul/quant/encode_kernels.cu`).
//!
//! The launchers are compiled by nvcc to an object file and linked in
//! (patrón zig-cuda-agent, igual que kernels/dequant_*.cu): aquí solo
//! declaramos los extern "C" y exponemos una API tipada.
//!
//! Contrato de bit-paridad: encodeMXFP4/encodeQ8_0 (src/kv_cache/kv_quant.zig)
//! son el oráculo — los kernels GPU son espejo bit-idéntico (ver
//! tests/test_quant_encode_gpu.zig).
const std = @import("std");
const debugz = @import("debug");
const build_options = @import("build_options");
const cudaz = @import("cudaz");

// Launchers nvcc (firma Runtime API; stream driver reinterpretado).
extern "c" fn quant_mxfp4_launcher(src: [*c]const f32, dst: [*c]u8, num_blocks: c_int, stream: cudaz.CUstream) void;
extern "c" fn quant_q8_0_launcher(src: [*c]const f32, dst: [*c]u8, num_blocks: c_int, stream: cudaz.CUstream) void;

pub const QuantEncodeError = error{
    CudaUnavailable,
    InvalidArgument,
};

/// Codifica un tensor f32 device a MXFP4 (17B por bloque de 32 elems).
/// `num_blocks` = elems/32; el buffer destino debe tener num_blocks*17 bytes.
pub fn encodeMxfp4(
    d_src: cudaz.CUdeviceptr,
    d_dst: cudaz.CUdeviceptr,
    num_blocks: usize,
    stream: cudaz.CUstream,
) QuantEncodeError!void {
    if (!build_options.has_cuda) return error.CudaUnavailable;
    if (num_blocks == 0) return;
    debugz.dbg.printLevel(.trace, "[quant_encode] mxfp4 blocks={d} src=0x{x} dst=0x{x}\n", .{ num_blocks, d_src, d_dst });
    quant_mxfp4_launcher(@ptrFromInt(d_src), @ptrFromInt(d_dst), @intCast(num_blocks), stream);
}

/// Codifica un tensor f32 device a Q8_0 (34B por bloque de 32 elems).
/// `num_blocks` = elems/32; el buffer destino debe tener num_blocks*34 bytes.
pub fn encodeQ8_0(
    d_src: cudaz.CUdeviceptr,
    d_dst: cudaz.CUdeviceptr,
    num_blocks: usize,
    stream: cudaz.CUstream,
) QuantEncodeError!void {
    if (!build_options.has_cuda) return error.CudaUnavailable;
    if (num_blocks == 0) return;
    debugz.dbg.printLevel(.trace, "[quant_encode] q8_0 blocks={d} src=0x{x} dst=0x{x}\n", .{ num_blocks, d_src, d_dst });
    quant_q8_0_launcher(@ptrFromInt(d_src), @ptrFromInt(d_dst), @intCast(num_blocks), stream);
}

/// Bytes de salida MXFP4 para `elems` elementos de entrada.
pub fn mxfp4Bytes(elems: usize) usize {
    return (elems / 32) * 17;
}

/// Bytes de salida Q8_0 para `elems` elementos de entrada.
pub fn q8_0Bytes(elems: usize) usize {
    return (elems / 32) * 34;
}
