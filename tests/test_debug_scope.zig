//! Tests de `Debug.scopeMatches` y filtro `DEBUG_SCOPE`.
const std = @import("std");
const debugz = @import("debug");

test "scopeMatches: sin scope activo, siempre true" {
    var d = debugz.Debug{};
    try std.testing.expect(d.scopeMatches("any"));
}

test "scopeMatches: tag activo, true" {
    var d = debugz.Debug{};
    d.scope_count = 1;
    const kv = "kv";
    @memcpy(d.scope_set[0][0..kv.len], kv);
    d.scope_set[0][kv.len] = 0;
    try std.testing.expect(d.scopeMatches("kv"));
}

test "scopeMatches: tag inactivo, false" {
    var d = debugz.Debug{};
    d.scope_count = 1;
    const kv = "kv";
    @memcpy(d.scope_set[0][0..kv.len], kv);
    d.scope_set[0][kv.len] = 0;
    try std.testing.expect(!d.scopeMatches("ssm"));
}

test "scopeMatches: multiple scopes" {
    var d = debugz.Debug{};
    d.scope_count = 2;
    const kv = "kv";
    const ssm = "ssm";
    @memcpy(d.scope_set[0][0..kv.len], kv);
    d.scope_set[0][kv.len] = 0;
    @memcpy(d.scope_set[1][0..ssm.len], ssm);
    d.scope_set[1][ssm.len] = 0;
    try std.testing.expect(d.scopeMatches("kv"));
    try std.testing.expect(d.scopeMatches("ssm"));
    try std.testing.expect(!d.scopeMatches("attn"));
}

test "scopeMatches: tag vacío, true" {
    var d = debugz.Debug{};
    d.scope_count = 1;
    const kv = "kv";
    @memcpy(d.scope_set[0][0..kv.len], kv);
    d.scope_set[0][kv.len] = 0;
    try std.testing.expect(d.scopeMatches(""));
}
