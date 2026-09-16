//! Chat template renderer para OpenAI / Anthropic / Ollama.
//!
//! Renderiza una lista de mensajes `{role, content}` al formato que el modelo
//! espera. Soporta:
//!   - Qwen/Qwen2/Qwen3 (ChatML: `<|im_start|>role\n...<|im_end|>\n`)
//!   - Llama-3 (`<|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>`)
//!   - Raw (sin template; útil para /v1/completions)
//!
//! Detección automática por el nombre del modelo o override vía `template_name`.
//! El GGUF del modelo ya trae su propio chat_template (jinja2-like) en
//! `gguf_tokenizer.applyChatTemplate`, pero ese sólo maneja system+user simple.
//! Aquí soportamos multi-turn (assistant, tool, system) y tool calls.

const std = @import("std");

pub const Role = enum {
    system,
    user,
    assistant,
    tool,
};

pub const Message = struct {
    role: Role,
    content: []const u8,
    /// Para role=tool: nombre de la función llamada (sólo se serializa
    /// en metadata, no en el prompt por ahora).
    name: ?[]const u8 = null,
};

/// Familia de template según el modelo. Detectado por el nombre o pasado
/// explícitamente por el caller.
pub const TemplateKind = enum {
    chatml, // Qwen, Qwen2, Qwen3, Qwen3.5, Qwen-VL, ...
    llama3, // Llama-3.1, Llama-3.2, ...
    raw, // Sin template (text completion)
};

/// Auto-detecta el template a partir del nombre de archivo del modelo.
/// Heurística simple (case-insensitive). Devuelve `raw` por defecto.
pub fn detectTemplateKind(model_name: []const u8) TemplateKind {
    var lower_buf: [256]u8 = undefined;
    const lower = if (model_name.len <= lower_buf.len)
        std.ascii.lowerString(&lower_buf, model_name)
    else
        std.ascii.lowerString(lower_buf[0..], model_name);
    if (std.mem.indexOf(u8, lower, "qwen") != null) return .chatml;
    if (std.mem.indexOf(u8, lower, "llama-3") != null or
        std.mem.indexOf(u8, lower, "llama3") != null) return .llama3;
    if (std.mem.indexOf(u8, lower, "mistral") != null) return .chatml; // Mistral-Instruct v0.3+ usa ChatML-like
    return .raw;
}

/// Renderiza la lista de mensajes al prompt que el modelo entrenó a esperar.
/// Caller owns the returned slice.
pub fn render(
    allocator: std.mem.Allocator,
    kind: TemplateKind,
    messages: []const Message,
) ![]u8 {
    return switch (kind) {
        .chatml => renderChatml(allocator, messages),
        .llama3 => renderLlama3(allocator, messages),
        .raw => renderRaw(allocator, messages),
    };
}

fn renderChatml(allocator: std.mem.Allocator, messages: []const Message) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (messages) |msg| {
        const role_str: []const u8 = switch (msg.role) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
            .tool => "user", // tool messages van como user en ChatML
        };
        try out.appendSlice(allocator, "<|im_start|>");
        try out.appendSlice(allocator, role_str);
        try out.append(allocator, '\n');
        try out.appendSlice(allocator, msg.content);
        try out.appendSlice(allocator, "<|im_end|>\n");
    }
    // Prefijo de turno assistant para que el modelo sepa que tiene que generar
    try out.appendSlice(allocator, "<|im_start|>assistant\n");
    return out.toOwnedSlice(allocator);
}

fn renderLlama3(allocator: std.mem.Allocator, messages: []const Message) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    // Header de BOS (Llama-3)
    try out.append(allocator, '<');
    try out.append(allocator, '|');
    try out.append(allocator, 'b');
    try out.append(allocator, 'e');
    try out.append(allocator, 'g');
    try out.append(allocator, '_');
    try out.append(allocator, 'o');
    try out.append(allocator, 'f');
    try out.append(allocator, '_');
    try out.append(allocator, 't');
    try out.append(allocator, 'e');
    try out.append(allocator, 'x');
    try out.append(allocator, 't');
    try out.append(allocator, '|');
    try out.append(allocator, '>');

    for (messages) |msg| {
        const role_str: []const u8 = switch (msg.role) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
            .tool => "ipython", // Llama-3 usa 'ipython' para tool results
        };
        try out.appendSlice(allocator, "<|start_header_id|>");
        try out.appendSlice(allocator, role_str);
        try out.appendSlice(allocator, "<|end_header_id|>\n\n");
        try out.appendSlice(allocator, msg.content);
        try out.appendSlice(allocator, "<|eot_id|>");
    }
    // Prefijo assistant
    try out.appendSlice(allocator, "<|start_header_id|>assistant<|end_header_id|>\n\n");
    return out.toOwnedSlice(allocator);
}

fn renderRaw(allocator: std.mem.Allocator, messages: []const Message) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (messages) |msg| {
        if (msg.role == .system) {
            try out.appendSlice(allocator, msg.content);
            try out.appendSlice(allocator, "\n\n");
        } else {
            try out.appendSlice(allocator, msg.content);
            try out.append(allocator, '\n');
        }
    }
    return out.toOwnedSlice(allocator);
}

// ─── Tests ───────────────────────────────────────────────────────────────

test "chatml renders Qwen turn sequence" {
    const allocator = std.testing.allocator;
    const msgs = [_]Message{
        .{ .role = .system, .content = "You are helpful." },
        .{ .role = .user, .content = "Hi" },
    };
    const out = try render(allocator, .chatml, &msgs);
    defer allocator.free(out);
    const expected =
        "<|im_start|>system\nYou are helpful.<|im_end|>\n" ++
        "<|im_start|>user\nHi<|im_end|>\n" ++
        "<|im_start|>assistant\n";
    try std.testing.expectEqualStrings(expected, out);
}

test "llama3 renders multi-turn" {
    const allocator = std.testing.allocator;
    const msgs = [_]Message{
        .{ .role = .system, .content = "sys" },
        .{ .role = .user, .content = "u" },
        .{ .role = .assistant, .content = "a" },
        .{ .role = .user, .content = "u2" },
    };
    const out = try render(allocator, .llama3, &msgs);
    defer allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "<|start_header_id|>assistant<|end_header_id|>\n\n") != null);
    try std.testing.expect(std.mem.endsWith(u8, out, "<|start_header_id|>assistant<|end_header_id|>\n\n"));
}

test "detect picks chatml for Qwen" {
    try std.testing.expectEqual(TemplateKind.chatml, detectTemplateKind("Qwen3-0.6B-Instruct"));
    try std.testing.expectEqual(TemplateKind.chatml, detectTemplateKind("qwen2.5-7b"));
    try std.testing.expectEqual(TemplateKind.llama3, detectTemplateKind("Llama-3.1-8B"));
    try std.testing.expectEqual(TemplateKind.raw, detectTemplateKind("unknown-7B"));
}
