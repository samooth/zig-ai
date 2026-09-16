//! Tests lane-server-f2 — regresión de los módulos del server.
//!
//! Ejecuta los test blocks inline de: auth, validation, audit, batching,
//! token_sink, quite — vía refAllDecls. NO requiere modelo ni GPU.

const std = @import("std");

test {
    _ = @import("auth");
    _ = @import("validation");
    _ = @import("audit");
    _ = @import("batching");
    _ = @import("token_sink");
    _ = @import("quite");
}
