//! Validation — bounds estrictos de TODOS los params de los endpoints.
//!
//! T4 del plan server F2. Un solo lugar con los límites; los handlers
//! llaman `validate.*` y devuelven el error JSON del protocolo si falla.
//! Los mensajes de error son estables (testeados) y SIN eco de contenido
//! del usuario (no reflejar el valor malicioso en la response).

const std = @import("std");

pub const ValidationError = error{
    TemperatureOutOfRange,
    TopPOutOfRange,
    TopKOutOfRange,
    PenaltiesOutOfRange,
    MaxTokensOutOfRange,
    MaxTokensExceedsContext,
    NGreaterThanOne,
    TooManyMessages,
    MessageTooLarge,
    TooManyStopSequences,
    StopSequenceTooLong,
    TooManyTools,
    ToolSchemaTooLarge,
    EmptyModel,
    InvalidStreamFlag,
    InvalidFinishReason,
};

/// Límites (constantes documentadas en API_SERVER.md).
pub const Limits = struct {
    pub const temperature_range: [2]f32 = .{ 0.0, 2.0 };
    pub const top_p_range: [2]f32 = .{ 0.0, 1.0 };
    pub const top_k_max: usize = 1000;
    pub const penalty_range: [2]f32 = .{ -2.0, 2.0 };
    pub const max_messages: usize = 128;
    pub const message_max_bytes: usize = 1024 * 1024; // 1 MiB
    pub const max_stop_sequences: usize = 4;
    pub const stop_seq_max_chars: usize = 64;
    pub const max_tools: usize = 32;
    pub const tool_schema_max_bytes: usize = 64 * 1024;
};

pub fn temperature(t: ?f32) ValidationError!f32 {
    const v = t orelse return 1.0;
    if (v < Limits.temperature_range[0] or v > Limits.temperature_range[1])
        return ValidationError.TemperatureOutOfRange;
    return v;
}

pub fn topP(p: ?f32) ValidationError!f32 {
    const v = p orelse return 1.0;
    if (v <= Limits.top_p_range[0] or v > Limits.top_p_range[1])
        return ValidationError.TopPOutOfRange;
    return v;
}

pub fn topK(k: ?usize) ValidationError!usize {
    const v = k orelse return 0;
    if (v > Limits.top_k_max) return ValidationError.TopKOutOfRange;
    return v;
}

pub fn penalties(presence: ?f32, frequency: ?f32) ValidationError!struct { p: f32, f: f32 } {
    const p = presence orelse 0;
    const f = frequency orelse 0;
    if (p < Limits.penalty_range[0] or p > Limits.penalty_range[1] or
        f < Limits.penalty_range[0] or f > Limits.penalty_range[1])
        return ValidationError.PenaltiesOutOfRange;
    return .{ .p = p, .f = f };
}

/// max_tokens + prompt_len contra context_length.
pub fn maxTokens(
    requested: ?usize,
    prompt_tokens: usize,
    context_length: usize,
) ValidationError!usize {
    const v = requested orelse return 256;
    if (v == 0 or v > 100_000) return ValidationError.MaxTokensOutOfRange;
    if (prompt_tokens + v > context_length) return ValidationError.MaxTokensExceedsContext;
    return v;
}

pub fn nSequences(n: ?usize) ValidationError!usize {
    const v = n orelse return 1;
    if (v != 1) return ValidationError.NGreaterThanOne;
    return v;
}

pub fn messages(count: usize) ValidationError!void {
    if (count == 0 or count > Limits.max_messages) return ValidationError.TooManyMessages;
}

pub fn messageContent(content: []const u8) ValidationError!void {
    if (content.len > Limits.message_max_bytes) return ValidationError.MessageTooLarge;
}

pub fn stopSequences(stops: []const []const u8) ValidationError!void {
    if (stops.len > Limits.max_stop_sequences) return ValidationError.TooManyStopSequences;
    for (stops) |s| {
        if (s.len > Limits.stop_seq_max_chars) return ValidationError.StopSequenceTooLong;
    }
}

pub fn tools(count: usize, largest_schema_bytes: usize) ValidationError!void {
    if (count > Limits.max_tools) return ValidationError.TooManyTools;
    if (largest_schema_bytes > Limits.tool_schema_max_bytes) return ValidationError.ToolSchemaTooLarge;
}

pub fn modelName(name: []const u8) ValidationError!void {
    if (name.len == 0) return ValidationError.EmptyModel;
}

// ─── Tests ───────────────────────────────────────────────────────────────

test "temperature bounds" {
    try std.testing.expectEqual(@as(f32, 0.7), try temperature(0.7));
    try std.testing.expectEqual(@as(f32, 1.0), try temperature(null));
    try std.testing.expectError(ValidationError.TemperatureOutOfRange, temperature(2.5));
    try std.testing.expectError(ValidationError.TemperatureOutOfRange, temperature(-0.1));
}

test "topP bounds" {
    try std.testing.expectEqual(@as(f32, 0.95), try topP(0.95));
    try std.testing.expectError(ValidationError.TopPOutOfRange, topP(0.0));
    try std.testing.expectError(ValidationError.TopPOutOfRange, topP(1.5));
}

test "maxTokens against context" {
    try std.testing.expectEqual(@as(usize, 128), try maxTokens(128, 10, 4096));
    try std.testing.expectError(ValidationError.MaxTokensExceedsContext, maxTokens(4000, 100, 4096));
    try std.testing.expectError(ValidationError.MaxTokensOutOfRange, maxTokens(0, 1, 100));
}

test "messages count and size" {
    try std.testing.expectError(ValidationError.TooManyMessages, messages(0));
    try messages(128);
    try std.testing.expectError(ValidationError.TooManyMessages, messages(129));
    const big = "x" ** (1024 * 1024 + 1);
    try std.testing.expectError(ValidationError.MessageTooLarge, messageContent(big));
}

test "stop sequences" {
    var ok = [_][]const u8{ "END", "\n\n" };
    try stopSequences(&ok);
    var many = [_][]const u8{ "a", "b", "c", "d", "e" };
    try std.testing.expectError(ValidationError.TooManyStopSequences, stopSequences(&many));
    var long = [_][]const u8{"x" ** 65};
    try std.testing.expectError(ValidationError.StopSequenceTooLong, stopSequences(&long));
}
