//! cudaz_ext_mem — bindings Driver API de memoria externa (Lane D).
//!
//! Archivo NUEVO y ajeno a cudaz_stub.zig (regla de lanes: no tocar el stub
//! compartido). Cubre cuMemHostRegister/Unregister: convertir regiones host
//! ya materializadas (p.ej. mmaps GGUF) en memoria pinned fuente válido para
//! H2D async / lectura directa desde kernels (UVA Linux).
//!
//! Lección HANDOFFS (incidente extern "c" meminfo): todo extern declarado
//! aquí está IMPLEMENTADO y usado por src/mem/host_bank.zig en el mismo commit.
const std = @import("std");
const debugz = @import("debug");

/// Flags de cuMemHostRegister (CUDA driver_api.h).
pub const CU_MEMHOSTREGISTER_PORTABLE: u32 = 0x01;
pub const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 0x02;
pub const CU_MEMHOSTREGISTER_NOMAP: u32 = 0x04;
pub const CU_MEMHOSTREGISTER_IO_MEMORY: u32 = 0x08;

/// Códigos de error específicos que este módulo puede producir (los comunes
/// viven en cudaz.CUresult; estos no existen aún allí y NO se añaden sin ticket).
pub const ERROR_HOST_MEMORY_ALREADY_REGISTERED: c_int = 712;
pub const ERROR_HOST_MEMORY_NOT_REGISTERED: c_int = 713;

const cudalib = struct {
    extern "c" fn cuMemHostRegister_v2(ptr: *anyopaque, bytes: usize, flags: u32) c_int;
    extern "c" fn cuMemHostUnregister(ptr: *anyopaque) c_int;
    /// Espera de un stream a un evento (cross-stream, base del doble búfer D5).
    /// No está en cudaz_stub: bindings propios de Lane D.
    extern "c" fn cuStreamWaitEvent(stream: *anyopaque, event: *anyopaque, flags: u32) c_int;
};

/// Tipos opacos locales coherentes con los handles del driver (CUstream/
/// CUevent son punteros opacos; cudaz los define igual pero este módulo no
/// depende de él para no arrastrar el grafo completo donde no hace falta).
pub fn streamWaitEvent(stream: *anyopaque, event: *anyopaque, flags: u32) !void {
    const res = cudalib.cuStreamWaitEvent(stream, event, flags);
    if (res != 0) return errorFrom(res);
}

/// Nombre legible de un CUresult numérico para breadcrumbs (subset relevante).
pub fn resultName(res: c_int) []const u8 {
    return switch (res) {
        0 => "SUCCESS",
        1 => "ERROR_INVALID_VALUE",
        2 => "ERROR_OUT_OF_MEMORY",
        3 => "ERROR_NOT_INITIALIZED",
        201 => "ERROR_INVALID_CONTEXT",
        208 => "ERROR_ALREADY_MAPPED",
        304 => "ERROR_OPERATING_SYSTEM",
        ERROR_HOST_MEMORY_ALREADY_REGISTERED => "ERROR_HOST_MEMORY_ALREADY_REGISTERED",
        ERROR_HOST_MEMORY_NOT_REGISTERED => "ERROR_HOST_MEMORY_NOT_REGISTERED",
        else => "UNKNOWN",
    };
}

/// Registra [ptr, ptr+bytes) como pinned. Requiere:
///   - ptr alineado a página host (4 KiB basta),
///   - bytes múltiplo de tamaño de página,
///   - contexto CUDA current (ver cudaz.ensureCurrent antes).
/// `flags`: DEVICEMAP hace la región desreferenciable desde kernels (UVA);
/// necesario para gathers futuros (Contrato 5/P2), inofensivo para memcpy.
pub fn hostRegister(ptr: *anyopaque, bytes: usize, flags: u32) !void {
    const res = cudalib.cuMemHostRegister_v2(ptr, bytes, flags);
    if (res != 0) return errorFrom(res);
}

/// Deshace hostRegister. Con bytes no registrados devuelve error.NotRegistered.
pub fn hostUnregister(ptr: *anyopaque) !void {
    const res = cudalib.cuMemHostUnregister(ptr);
    if (res != 0) return errorFrom(res);
}

/// Espera `event` en `stream` (cross-stream sync para el doble búfer
/// copy↔compute, Contrato 5 / Lane D D3). flags suele ser 0.
/// (Definido arriba junto al struct cudalib — desduplicado tras merge.)
fn errorFrom(res: c_int) error{ AlreadyRegistered, NotRegistered, InvalidValue, OutOfMemory, NotInitialized, InvalidContext, CudaError } {
    // Breadcrumb con código numérico antes de perderlo en el error tag.
    debugz.dbg.print("[gpu_kernels] fallo driver: {s} ({d})\n", .{ resultName(res), res });
    return switch (res) {
        ERROR_HOST_MEMORY_ALREADY_REGISTERED, 208 => error.AlreadyRegistered,
        ERROR_HOST_MEMORY_NOT_REGISTERED, 211 => error.NotRegistered,
        1 => error.InvalidValue,
        2 => error.OutOfMemory,
        3 => error.NotInitialized,
        201 => error.InvalidContext,
        else => error.CudaError,
    };
}

test "resultName cubre códigos nuevos" {
    try std.testing.expectEqualStrings("ERROR_HOST_MEMORY_ALREADY_REGISTERED", resultName(712));
    try std.testing.expectEqualStrings("SUCCESS", resultName(0));
    try std.testing.expectEqualStrings("UNKNOWN", resultName(12345));
}
