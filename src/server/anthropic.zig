//! Anthropic Messages API v2: /v1/messages, /v1/messages/count_tokens.
//!
//! T6 del plan server F2. Reemplaza el pseudo-streaming de Fase 1:
//! el engine corre IN-PROCESS (runHybridInferenceSink) y cada token
//! emitido se serializa como evento SSE Anthropic al vuelo.
//!
//! Schema fiel a https://docs.anthropic.com/en/api/messages:
//!   POST /v1/messages
//!     → x-api-key auth (mod.zig) + anthropic-version header
//!     → system (separado) + messages → chat_template.render
//!     → SseSink(TokenSink) sobre ctx.startSSEStreaming()
//!       event: message_start / content_block_start / ping /
//!              content_block_delta (text_delta por token) /
//!              content_block_stop / message_delta / message_stop
//!
//! No-stream: CollectingSink + response JSON única con usage.
//!
//! count_tokens: BPE REAL del modelo (no heurística chars/4).

const std = @import("std");
const httpx = @import("httpx");
const sse_mod = @import("sse");
const chat_template = @import("chat_template");
const inference = @import("inference");
const bpe = @import("tokenizer");
const gguf_tokenizer = @import("gguf_tokenizer");
const time_mod = @import("time");
const json_util = @import("json_util");
const token_sink = @import("token_sink");
const validation = @import("validation");
const quiet = @import("quite");

const TokenSink = token_sink.TokenSink;
const CollectingSink = token_sink.CollectingSink;

const jsonStringify = json_util.jsonStringify;
const rand_u64 = json_util.rand_u64;
const Message = chat_template.Message;
const Role = chat_template.Role;

fn nowSec() i64 {
    return time_mod.wallClockSec();
}

// ─── Estado del server (compartido con openai.zig vía mod.zig) ─────────────

const openai = @import("openai");
const ServerState = openai.ServerState;

var g_state: ?*ServerState = null;

pub fn setState(st: *ServerState) void {
    g_state = st;
}

fn getState() *ServerState {
    return g_state orelse unreachable;
}

// ─── Request schema ───────────────────────────────────────────────────────

const MessageJson = struct {
    role: []const u8,
    content: []const u8,
};

const MessagesRequest = struct {
    model: []const u8,
    messages: []MessageJson,
    system: ?[]const u8 = null,
    max_tokens: usize = 256,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?usize = null,
    stream: ?bool = null,
    stop_sequences: ?[][]const u8 = null,
    stop_sequence: ?[]const u8 = null,
    metadata: ?std.json.Value = null,
};

// ─── Error responses (formato Anthropic) ───────────────────────────────────

fn errorResponse(ctx: *httpx.Context, status: u16, err_type: []const u8, msg: []const u8) !httpx.Response {
    const body = try jsonStringify(ctx.allocator, .{
        .type = "error",
        .err = .{
            .type = err_type,
            .message = msg,
        },
    });
    return httpx.Response.fromJson(ctx.allocator, status, body);
}

// ─── POST /v1/messages ─────────────────────────────────────────────────────

pub fn messages(ctx: *httpx.Context) !httpx.Response {
    const st = getState();

    // 1. Body + parse.
    const body = ctx.request.body orelse
        return errorResponse(ctx, 400, "invalid_request_error", "missing body");
    const parsed = std.json.parseFromSlice(MessagesRequest, ctx.allocator, body, .{
        .ignore_unknown_fields = true,
    }) catch
        return errorResponse(ctx, 400, "invalid_request_error", "invalid JSON");
    defer parsed.deinit();
    const req = parsed.value;

    // 2. Validation (T4).
    validation.modelName(req.model) catch
        return errorResponse(ctx, 400, "invalid_request_error", "model must be non-empty");
    validation.messages(req.messages.len) catch
        return errorResponse(ctx, 400, "invalid_request_error", "messages: 1..128 items");
    if (req.system) |sys| validation.messageContent(sys) catch
        return errorResponse(ctx, 400, "invalid_request_error", "system content exceeds 1 MiB");
    for (req.messages) |m| validation.messageContent(m.content) catch
        return errorResponse(ctx, 400, "invalid_request_error", "message content exceeds 1 MiB");
    const temp = validation.temperature(req.temperature) catch
        return errorResponse(ctx, 400, "invalid_request_error", "temperature must be in [0,2]");
    const top_p = validation.topP(req.top_p) catch
        return errorResponse(ctx, 400, "invalid_request_error", "top_p must be in (0,1]");
    const top_k = validation.topK(req.top_k) catch
        return errorResponse(ctx, 400, "invalid_request_error", "top_k too large");

    // stop_sequences: máximo 4 de 64 chars (mismo límite OpenAI).
    var stops_buf: [4][]const u8 = undefined;
    var stops_len: usize = 0;
    if (req.stop_sequence) |s| {
        stops_buf[0] = s;
        stops_len = 1;
    }
    if (req.stop_sequences) |multi| {
        stops_len = @min(multi.len, stops_buf.len);
        for (multi[0..stops_len], 0..) |s, i| stops_buf[i] = s;
    }
    validation.stopSequences(stops_buf[0..stops_len]) catch
        return errorResponse(ctx, 400, "invalid_request_error", "stop_sequences: max 4 of 64 chars");

    // 3. system (separado en Anthropic) + messages → chat template.
    var msgs: std.ArrayList(Message) = .empty;
    defer msgs.deinit(ctx.allocator);
    if (req.system) |sys| {
        try msgs.append(ctx.allocator, .{ .role = .system, .content = sys });
    }
    for (req.messages) |m| {
        const role: Role = if (std.mem.eql(u8, m.role, "user"))
            .user
        else if (std.mem.eql(u8, m.role, "assistant"))
            .assistant
        else
            return errorResponse(ctx, 400, "invalid_request_error", "only user/assistant roles allowed");
        try msgs.append(ctx.allocator, .{ .role = role, .content = m.content });
    }
    const kind = chat_template.detectTemplateKind(st.model_id);
    const prompt = try chat_template.render(ctx.allocator, kind, msgs.items);
    defer ctx.allocator.free(prompt);

    // max_tokens vs context (est conservador chars/4).
    const prompt_est_tokens = @divFloor(prompt.len, 4);
    const max_tokens = validation.maxTokens(req.max_tokens, prompt_est_tokens, st.context_length) catch
        return errorResponse(ctx, 400, "invalid_request_error", "max_tokens exceeds context window");

    // 4. Params del engine.
    var cli_params = st.backend;
    cli_params.prompt = prompt;
    cli_params.max_new_tokens = max_tokens;
    cli_params.sampler = .{
        .temperature = temp,
        .top_p = top_p,
        .top_k = top_k,
    };
    cli_params.seed = 42;

    // 5. Dispatch: streaming vs blocking.
    const stops: []const []const u8 = stops_buf[0..stops_len];
    if (req.stream orelse false) {
        return streamMessages(ctx, st, cli_params, req.model, stops);
    } else {
        return blockMessages(ctx, st, cli_params, req.model, stops);
    }
}

// ─── Camino no-stream ─────────────────────────────────────────────────────

fn blockMessages(
    ctx: *httpx.Context,
    st: *ServerState,
    cli_params: inference.CliParams,
    model_name: []const u8,
    stops: []const []const u8,
) !httpx.Response {
    var sink_ctx = CollectingSink.init(ctx.allocator);
    defer sink_ctx.deinit();
    var stops_arr: [4][]const u8 = undefined;
    copyStops(&stops_arr, stops);
    sink_ctx.stop_seqs = stops_arr[0..stops.len];

    openai.runEnginePub(st, cli_params, sink_ctx.sink()) catch |e| {
        return errorResponse(ctx, 500, "api_error", @errorName(e));
    };

    const usage: token_sink.Usage = sink_ctx.usage orelse .{
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .total_ms = 0,
    };
    const stop_reason: []const u8 = if (sink_ctx.finish_reason == .length) "max_tokens" else "end_turn";

    const id = try std.fmt.allocPrint(ctx.allocator, "msg_{x}", .{rand_u64()});
    defer ctx.allocator.free(id);

    return ctx.json(.{
        .id = id,
        .type = "message",
        .role = "assistant",
        .model = model_name,
        .content = .{
            .{
                .type = "text",
                .text = sink_ctx.text.items,
            },
        },
        .stop_reason = stop_reason,
        .stop_sequence = null,
        .usage = .{
            .input_tokens = usage.prompt_tokens,
            .output_tokens = usage.completion_tokens,
        },
    });
}

// ─── Camino stream (SSE Anthropic real, token-por-token) ──────────────────

fn streamMessages(
    ctx: *httpx.Context,
    st: *ServerState,
    cli_params: inference.CliParams,
    model_name: []const u8,
    stops: []const []const u8,
) !httpx.Response {
    const writer = try ctx.startSSEStreaming();

    var sink_ctx = AnthropicSseSink.init(ctx.allocator, writer, model_name);
    defer sink_ctx.deinit();
    var stops_arr: [4][]const u8 = undefined;
    copyStops(&stops_arr, stops);
    sink_ctx.stop_seqs = stops_arr[0..stops.len];

    openai.runEnginePub(st, cli_params, sink_ctx.sink()) catch |e| {
        sink_ctx.emitError(@errorName(e));
        return httpx.Response.fromText(ctx.allocator, 200, "");
    };

    sink_ctx.emitTrailers();
    return httpx.Response.fromText(ctx.allocator, 200, "");
}

/// Escribe un evento SSE Anthropic: `event: <name>\ndata: <json>\n\n`.
fn writeEvent(writer: httpx.StreamWriter, event: []const u8, data: []const u8) !void {
    try writer.writeAll("event: ");
    try writer.writeAll(event);
    try writer.writeAll("\ndata: ");
    try writer.writeAll(data);
    try writer.writeAll("\n\n");
}

fn copyStops(dst: *[4][]const u8, src: []const []const u8) void {
    for (src, 0..) |s, i| dst[i] = s;
}

// ─── AnthropicSseSink: TokenSink → eventos SSE Anthropic ──────────────────

pub const AnthropicSseSink = struct {
    allocator: std.mem.Allocator,
    writer: httpx.StreamWriter,
    model_name: []const u8,
    id: []const u8,
    created: i64,
    /// Stop-sequences (detección incremental, corte al vuelo).
    stop_seqs: []const []const u8 = &.{},
    /// Texto acumulado (para detección incremental de stops).
    acc: std.ArrayList(u8) = .empty,
    usage: ?token_sink.Usage = null,
    finish_reason: token_sink.FinishReason = .stop,
    /// message_start emitido (input_tokens del prefill).
    input_tokens: usize = 0,
    /// message_start + content_block_start + ping ya emitidos.
    message_started: bool = false,
    finished: bool = false,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, writer: httpx.StreamWriter, model_name: []const u8) Self {
        const id = std.fmt.allocPrint(allocator, "msg_{x}", .{rand_u64()}) catch "msg_err";
        return .{
            .allocator = allocator,
            .writer = writer,
            .model_name = model_name,
            .id = id,
            .created = nowSec(),
        };
    }

    pub fn deinit(self: *Self) void {
        if (!std.mem.eql(u8, self.id, "msg_err")) self.allocator.free(self.id);
        self.acc.deinit(self.allocator);
    }

    pub fn sink(self: *Self) TokenSink {
        return .{ .ctx = self, .vtable = &vtable };
    }

    const vtable = TokenSink.VTable{
        .emit = emitImpl,
        .finish = finishImpl,
    };

    fn emitImpl(ctx: *anyopaque, token: u32, text: []const u8) token_sink.SinkSignal!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        _ = token;

        // Primera pieza: message_start + content_block_start + ping.
        if (!self.message_started) {
            self.emitStart() catch {};
            self.message_started = true;
        }

        // Detección incremental de stop-sequences: si el texto acumulado
        // termina en un prefijo propio de un stop, emitir y cortar en
        // el finish (strip del texto no emitido).
        self.acc.appendSlice(self.allocator, text) catch {};
        if (self.hitStop()) return token_sink.SinkSignal.StopSequenceHit;

        // content_block_delta por token.
        const data = self.jsonDeltaText(text) catch return;
        writeEvent(self.writer, "content_block_delta", data) catch {};
        self.allocator.free(data);
    }

    fn finishImpl(ctx: *anyopaque, reason: token_sink.FinishReason, usage: token_sink.Usage) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.finish_reason = reason;
        self.usage = usage;
        self.input_tokens = usage.prompt_tokens;
    }

    /// ¿El texto acumulado termina en alguna stop-sequence?
    fn hitStop(self: *Self) bool {
        for (self.stop_seqs) |ss| {
            if (ss.len > 0 and std.mem.endsWith(u8, self.acc.items, ss)) return true;
        }
        return false;
    }

    /// message_start + content_block_start + ping.
    fn emitStart(self: *Self) !void {
        const s1 = try jsonStringify(self.allocator, .{
            .type = "message_start",
            .message = .{
                .id = self.id,
                .type = "message",
                .role = "assistant",
                .model = self.model_name,
                .content = &[_]u8{},
                .stop_reason = null,
                .stop_sequence = null,
                .usage = .{ .input_tokens = 0, .output_tokens = 0 },
            },
        });
        defer self.allocator.free(s1);
        try writeEvent(self.writer, "message_start", s1);

        const s2 = try jsonStringify(self.allocator, .{
            .type = "content_block_start",
            .index = 0,
            .content_block = .{ .type = "text", .text = "" },
        });
        defer self.allocator.free(s2);
        try writeEvent(self.writer, "content_block_start", s2);

        try writeEvent(self.writer, "ping", "{\"type\":\"ping\"}");
    }

    /// content_block_delta text_delta.
    fn jsonDeltaText(self: *Self, text: []const u8) ![]u8 {
        return jsonStringify(self.allocator, .{
            .type = "content_block_delta",
            .index = 0,
            .delta = .{ .type = "text_delta", .text = text },
        });
    }

    /// Eventos de cierre: content_block_stop, message_delta, message_stop.
    /// Se llaman tras runEngine (streamMessages) o tras error.
    pub fn emitTrailers(self: *Self) void {
        const s1 = jsonStringify(self.allocator, .{
            .type = "content_block_stop",
            .index = 0,
        }) catch return;
        defer self.allocator.free(s1);
        writeEvent(self.writer, "content_block_stop", s1) catch {};

        const stop_reason: []const u8 = switch (self.finish_reason) {
            .length => "max_tokens",
            .stop => "end_turn",
            else => "end_turn",
        };
        const usage = self.usage orelse token_sink.Usage{
            .prompt_tokens = 0,
            .completion_tokens = 0,
            .total_ms = 0,
        };
        const s2 = jsonStringify(self.allocator, .{
            .type = "message_delta",
            .delta = .{ .stop_reason = stop_reason, .stop_sequence = null },
            .usage = .{ .output_tokens = usage.completion_tokens },
        }) catch return;
        defer self.allocator.free(s2);
        writeEvent(self.writer, "message_delta", s2) catch {};

        const s3 = jsonStringify(self.allocator, .{
            .type = "message_stop",
        }) catch return;
        defer self.allocator.free(s3);
        writeEvent(self.writer, "message_stop", s3) catch {};
    }

    /// Error mid-stream: event error (protocolo Anthropic).
    pub fn emitError(self: *Self, err_name: []const u8) void {
        const s = jsonStringify(self.allocator, .{
            .type = "error",
            .err = .{ .type = "api_error", .message = err_name },
        }) catch return;
        defer self.allocator.free(s);
        writeEvent(self.writer, "error", s) catch {};
    }
};

// ─── POST /v1/messages/count_tokens ───────────────────────────────────────

pub fn countTokens(ctx: *httpx.Context) !httpx.Response {
    const st = getState();
    const body = ctx.request.body orelse
        return errorResponse(ctx, 400, "invalid_request_error", "missing body");
    const parsed = std.json.parseFromSlice(MessagesRequest, ctx.allocator, body, .{
        .ignore_unknown_fields = true,
    }) catch
        return errorResponse(ctx, 400, "invalid_request_error", "invalid JSON");
    defer parsed.deinit();
    const req = parsed.value;

    // 1. Render del prompt (mismo camino que /v1/messages).
    var msgs: std.ArrayList(Message) = .empty;
    defer msgs.deinit(ctx.allocator);
    if (req.system) |sys| {
        try msgs.append(ctx.allocator, .{ .role = .system, .content = sys });
    }
    for (req.messages) |m| {
        const role: Role = if (std.mem.eql(u8, m.role, "user"))
            .user
        else if (std.mem.eql(u8, m.role, "assistant"))
            .assistant
        else
            return errorResponse(ctx, 400, "invalid_request_error", "only user/assistant roles allowed");
        try msgs.append(ctx.allocator, .{ .role = role, .content = m.content });
    }
    const kind = chat_template.detectTemplateKind(st.model_id);
    const prompt = try chat_template.render(ctx.allocator, kind, msgs.items);
    defer ctx.allocator.free(prompt);

    // 2. BPE REAL del modelo (no heurística chars/4).
    var gt = gguf_tokenizer.GgufTokenizer.fromGguf(ctx.allocator, &st.model.file) catch {
        // Fallback conservador si el tokenizer no carga.
        return ctx.json(.{ .input_tokens = @divFloor(prompt.len, 4) });
    };
    defer gt.deinit();
    var tok = bpe.BPETokenizer.fromTokenizer(ctx.allocator, &gt) catch {
        return ctx.json(.{ .input_tokens = @divFloor(prompt.len, 4) });
    };
    defer tok.deinit();
    const ids = tok.encode(prompt, .{}) catch {
        return ctx.json(.{ .input_tokens = @divFloor(prompt.len, 4) });
    };
    defer ctx.allocator.free(ids);

    return ctx.json(.{ .input_tokens = ids.len });
}

test "anthropic copyStops" {
    var arr: [4][]const u8 = undefined;
    const stops = [_][]const u8{ "a", "b" };
    copyStops(&arr, &stops);
    try std.testing.expectEqualStrings("a", arr[0]);
    try std.testing.expectEqualStrings("b", arr[1]);
}
