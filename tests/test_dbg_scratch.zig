//! Scratch test genérico para `zig build dbg`.
//!
//! Este archivo existe para iteración rápida de INFRA: edita el cuerpo de
//! `scratch()` y ejecuta `zig build dbg` para compilar/run solo este test.
//! No forma parte de la suite `zig build test`.
const std = @import("std");

test "dbg scratch" {
    try std.testing.expect(true);
}

// Coloca aquí código provisional de diagnóstico, prototipos de API debug,
// breadcrumbs, o asserts rápidos. Si lo necesitas, importa módulos del
// proyecto (ej: `@import("../../src/debug.zig")`) y úsalo como harness.
pub fn scratch() void {
    // TODO (INFRA): reemplazar por prueba concreta.
}
