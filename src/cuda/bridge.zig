//! UC-3.1 (TODO_CUDA.md, lane-cuda): bridge type-safe sobre
//! `cuModuleGetFunction` — el nombre del kernel es un ENUM comptime:
//! un typo (`getFunction(mod, .argmaxF32Kernell)`) = ERROR DE COMPILACIÓN,
//! no ERROR_NOT_FOUND en runtime. La clase de bug del ABI `int dbg` de
//! lane-a era exactamente esto (nombre/aridad mal cableados en silencio).
//!
//! Patrón adaptado del estudio del repo externo coderonion/zcuda (MIT):
//! src/kernel/bridge_gen.zig:84-88 — enum `Fn` construido comptime desde
//! `fn_names`. Atribución; cero dependencia.
//!
//! USO:
//!   const LK = bridge.Bridge(kernel_names);   // kernel_names: []const []const u8
//!   const f = try LK.getFunction(module, .rmsNormKernel); // typo = no compila
//!   // escape hatch runtime (strings dinámicos):
//!   const g = try LK.getFunctionByName(module, "rmsNormKernel");
const std = @import("std");
const cudaz = @import("cudaz");

/// Construye un namespace con:
///   - `Fn`: enum de nombres de kernel (comptime, typo-safe)
///   - `getFunction(mod, .name)`: CUfunction con check comptime
///   - `getFunctionByName(mod, name)`: escape hatch runtime
///   - `name(comptime f)`: []const u8 del tag (para tablas)
pub fn Bridge(comptime names: []const []const u8) type {
    if (names.len == 0) @compileError("bridge.Bridge: la lista de kernels no puede estar vacía");
    const Tag = std.math.IntFittingRange(0, names.len - 1);
    return struct {
        /// Enum comptime de los nombres. `@tagName(.x)` da el string exacto.
        /// Construido con `@Enum` (mismo patrón que la ref MIT).
        pub const Fn = @Enum(Tag, .exhaustive, names, &std.simd.iota(Tag, names.len));

        /// Nº de kernels de la tabla (comptime).
        pub const count: usize = names.len;

        /// CUfunction por enum comptime. Typo = compile error.
        pub fn getFunction(module: cudaz.CUmodule, comptime f: Fn) !cudaz.CUfunction {
            return cudaz.cuModuleGetFunction(module, @tagName(f));
        }

        /// Escape hatch por string (nombres dinámicos). Pierde el check
        /// comptime a propósito — úsalo sólo cuando el nombre no es
        /// conocido en compilación.
        pub fn getFunctionByName(module: cudaz.CUmodule, func_name: []const u8) !cudaz.CUfunction {
            return cudaz.cuModuleGetFunction(module, func_name);
        }

        /// ¿`name` está en la tabla? (comptime, p.ej. para asserts).
        pub fn contains(comptime name: []const u8) bool {
            inline for (names) |n| {
                if (std.mem.eql(u8, n, name)) return true;
            }
            return false;
        }
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests unitarios (sin GPU): el valor real es que los typos NO compilan; en
// runtime/compile-time verificamos la superficie del enum y `contains`.
// ─────────────────────────────────────────────────────────────────────────────

test "bridge: enum comptime con los nombres exactos" {
    const LK = Bridge(&.{ "rmsNormKernel", "argmaxF32Kernel", "qgemmKernel" });
    try std.testing.expectEqual(@as(usize, 3), LK.count);
    try std.testing.expectEqualStrings("rmsNormKernel", @tagName(LK.Fn.rmsNormKernel));
    try std.testing.expectEqualStrings("argmaxF32Kernel", @tagName(LK.Fn.argmaxF32Kernel));
    try std.testing.expect(LK.contains("qgemmKernel"));
    try std.testing.expect(!LK.contains("qgemmKernel_typo"));
}

test "bridge: enum con 1 elemento (tag_type u0) no rompe" {
    const One = Bridge(&.{"soloKernel"});
    try std.testing.expectEqualStrings("soloKernel", @tagName(One.Fn.soloKernel));
}
