//! OpenAI-compatible API v2 — streaming SSE REAL token-por-token.
//!
//! T5 del plan server F2. Reemplaza el pseudo-streaming de Fase 1:
//! el engine corre IN-PROCESS (runHybridInferenceSink con NullWriter
//! para banners) y cada token emitido se serializa como chunk SSE
//! `chat.completion.chunk` al vuelo.
//!
//! Flujo:
//!   POST /v1/chat/completions
//!     → auth (middleware mod.zig) → validation (T4)
//!     → chat_template.render → BPE encode
//!     → SseSink(TokenSink) sobre ctx.startSSEStreaming()
//!     → runHybridInferenceSink(..., sink) [token → chunk SSE]
//!     → [DONE]
//!
//! No-stream: CollectingSink + response JSON única con usage.

const std = @import("std");
const httpx = @import("httpx");
const sse_mod = @import("sse");
const chat_template = @import("chat_template");
const inference = @import("inference");
const bpe = @import("tokenizer");
const gguf_model = @import("gguf_model");
const time_mod = @import("time");
const json_util = @import("json_util");
const token_sink = @import("token_sink");
const validation = @import("validation");
const quiet = @import("quite");

const TokenSink = token_sink.TokenSink;
const CollectingSink = token_sink.CollectingSink;
const Engine = @import("engine").Engine;

const jsonStringify = json_util.jsonStringify;
const rand_u64 = json_util.rand_u64;
const Message = chat_template.Message;

fn nowSec() i64 {
    return time_mod.wallClockSec();
}

/// Estado global del server (seteado por mod.zig en runServer):
/// modelo cargado + config persistente.
pub const ServerState = struct {
    allocator: std.mem.Allocator,
    model: *gguf_model.GgufModel,
    model_path: []const u8,
    backend: inference.CliParams,
    model_id: []const u8,
    context_length: u32,
    architecture: []const u8,
};

var g_state: ?*ServerState = null;

pub fn setState(st: *ServerState) void {
    g_state = st;
}

fn getState() *ServerState {
    return g_state orelse unreachable;
}

// ─── Request schema (parcial; unknown fields ignorados) ───────────────────

const ChatRequest = struct {
    model: []const u8,
    messages: []MessageJson,
    max_tokens: ?usize = null,
    max_completion_tokens: ?usize = null,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?usize = null,
    n: ?usize = null,
    stream: ?bool = null,
    stream_options: ?StreamOptions = null,
    stop: ?StopSeq = null,
    presence_penalty: ?f32 = null,
    frequency_penalty: ?f32 = null,
    seed: ?u64 = null,

    const StreamOptions = struct {
        include_usage: bool = false,
    };
    const StopSeq = union(enum) {
        single: []const u8,
        multi: [][]const u8,
    };
};

const MessageJson = struct {
    role: []const u8,
    content: []const u8,
    name: ?[]const u8 = null,
};

// ─── Error responses (formato OpenAI) ─────────────────────────────────────

fn errorResponse(ctx: *httpx.Context, status: u16, err_type: []const u8, msg: []const u8) !httpx.Response {
    const body = try jsonStringify(ctx.allocator, .{
        .err = .{
            .message = msg,
            .type = err_type,
            .param = "",
            .code = "",
        },
    });
    return httpx.Response.fromJson(ctx.allocator, status, body);
}

// ─── POST /v1/chat/completions ────────────────────────────────────────────

pub fn chatCompletions(ctx: *httpx.Context) !httpx.Response {
    const st = getState();

    // 1. Body + parse (límite de tamaño vía httpx config en mod.zig).
    const body = ctx.request.body orelse
        return errorResponse(ctx, 400, "invalid_request_error", "missing body");
    const parsed = std.json.parseFromSlice(ChatRequest, ctx.allocator, body, .{
        .ignore_unknown_fields = true,
    }) catch
        return errorResponse(ctx, 400, "invalid_request_error", "invalid JSON");
    defer parsed.deinit();
    const req = parsed.value;

    // 2. Validation estricta (T4) — sin eco de contenido.
    validation.modelName(req.model) catch
        return errorResponse(ctx, 400, "invalid_request_error", "model must be non-empty");
    validation.messages(req.messages.len) catch
        return errorResponse(ctx, 400, "invalid_request_error", "messages: 1..128 items");
    for (req.messages) |m| validation.messageContent(m.content) catch
        return errorResponse(ctx, 400, "invalid_request_error", "message content exceeds 1 MiB");
    _ = validation.nSequences(req.n) catch
        return errorResponse(ctx, 400, "invalid_request_error", "n>1 not supported");
    const temp = validation.temperature(req.temperature) catch
        return errorResponse(ctx, 400, "invalid_request_error", "temperature must be in [0,2]");
    const top_p = validation.topP(req.top_p) catch
        return errorResponse(ctx, 400, "invalid_request_error", "top_p must be in (0,1]");
    const top_k = validation.topK(req.top_k) catch
        return errorResponse(ctx, 400, "invalid_request_error", "top_k too large");
    _ = validation.penalties(req.presence_penalty, req.frequency_penalty) catch
        return errorResponse(ctx, 400, "invalid_request_error", "penalties must be in [-2,2]");

    var stops_buf: [4][]const u8 = undefined;
    var stops_len: usize = 0;
    if (req.stop) |sv| switch (sv) {
        .single => |s| {
            stops_buf[0] = s;
            stops_len = 1;
        },
        .multi => |multi| {
            stops_len = @min(multi.len, stops_buf.len);
            for (multi[0..stops_len], 0..) |s, i| stops_buf[i] = s;
        },
    };
    validation.stopSequences(stops_buf[0..stops_len]) catch
        return errorResponse(ctx, 400, "invalid_request_error", "stop: max 4 sequences of 64 chars");

    // 3. Template + BPE (token count REAL para validación de context).
    var msgs: std.ArrayList(Message) = .empty;
    defer msgs.deinit(ctx.allocator);
    for (req.messages) |m| {
        const role: chat_template.Role = roleFromString(m.role) orelse
            return errorResponse(ctx, 400, "invalid_request_error", "invalid role");
        try msgs.append(ctx.allocator, .{ .role = role, .content = m.content, .name = m.name });
    }
    const kind = chat_template.detectTemplateKind(st.model_id);
    const prompt = try chat_template.render(ctx.allocator, kind, msgs.items);
    defer ctx.allocator.free(prompt);

    // BPE encode del prompt (via engine state; mismo tokenizer del modelo).
    // NOTA: runHybridInferenceSink hace su propio encode interno desde el
    // `prompt` string; el count aquí es para validar max_tokens vs ctx.
    // Heurística conservadora chars/4 para el pre-check (el engine trunca
    // con seguridad si excede).
    const prompt_est_tokens = @divFloor(prompt.len, 4);
    const max_tokens = validation.maxTokens(
        req.max_completion_tokens orelse req.max_tokens,
        prompt_est_tokens,
        st.context_length,
    ) catch
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
    cli_params.seed = req.seed orelse 42;

    // 5. Dispatch: streaming vs blocking.
    if (req.stream orelse false) {
        return streamChat(ctx, st, cli_params, req.model, req.stop, req.stream_options);
    } else {
        return blockChat(ctx, st, cli_params, req.model, req.stop);
    }
}

/// Camino no-stream: CollectingSink → response JSON única.
fn blockChat(
    ctx: *httpx.Context,
    st: *ServerState,
    cli_params: inference.CliParams,
    model_name: []const u8,
    stop: ?ChatRequest.StopSeq,
) !httpx.Response {
    var sink_ctx = CollectingSink.init(ctx.allocator);
    defer sink_ctx.deinit();
    // Buffer del frame de blockChat: vive durante runEngine (los strings
    // apuntan al JSON parsed, vivo hasta el final de chatCompletions).
    var stops_arr: [4][]const u8 = undefined;
    applyStops(&sink_ctx, stop, &stops_arr);

    runEngine(st, cli_params, sink_ctx.sink()) catch |e| {
        return errorResponse(ctx, 500, "server_error", @errorName(e));
    };

    const usage: token_sink.Usage = sink_ctx.usage orelse .{
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .total_ms = 0,
    };
    const finish: []const u8 = if (sink_ctx.finish_reason == .length) "length" else "stop";

    const id = try std.fmt.allocPrint(ctx.allocator, "chatcmpl-{x}", .{rand_u64()});
    defer ctx.allocator.free(id);

    return ctx.json(.{
        .id = id,
        .object = "chat.completion",
        .created = nowSec(),
        .model = model_name,
        .choices = .{
            .{
                .index = 0,
                .message = .{
                    .role = "assistant",
                    .content = sink_ctx.text.items,
                },
                .finish_reason = finish,
            },
        },
        .usage = .{
            .prompt_tokens = usage.prompt_tokens,
            .completion_tokens = usage.completion_tokens,
            .total_tokens = usage.prompt_tokens + usage.completion_tokens,
        },
    });
}

/// Camino stream: SseSink → chunk SSE por token + [DONE].
fn streamChat(
    ctx: *httpx.Context,
    st: *ServerState,
    cli_params: inference.CliParams,
    model_name: []const u8,
    stop: ?ChatRequest.StopSeq,
    stream_options: ?ChatRequest.StreamOptions,
) !httpx.Response {
    const writer = try ctx.startSSEStreaming();

    // SseSink: TokenSink que escribe chunks SSE al StreamWriter de httpx.
    var sink_ctx = SseSink.init(ctx.allocator, writer, model_name);
    defer sink_ctx.deinit();
    var stops_arr: [4][]const u8 = undefined;
    applyStopsSse(&sink_ctx, stop, &stops_arr);

    runEngine(st, cli_params, sink_ctx.sink()) catch |e| {
        // Error a mitad de stream: chunk de error + [DONE] (el status ya
        // es 200; el cliente lo ve como terminación anormal).
        sink_ctx.emitError(@errorName(e));
        try writer.writeAll("data: [DONE]\n\n");
        // SSE usa framing propio (no chunked): NO escribir terminator
        // chunked aquí. streaming_done=true hace que el server no
        // formatee nada más tras el handler.
        return httpx.Response.fromText(ctx.allocator, 200, "");
    };

    // Chunk final con finish_reason + usage opcional.
    try sink_ctx.emitFinish(stream_options orelse .{});
    try writer.writeAll("data: [DONE]\n\n");
    return httpx.Response.fromText(ctx.allocator, 200, "");
}

/// Ejecuta el engine in-process con sink. Este es el punto donde el
/// modelo corre de verdad (runHybridInferenceSink con NullWriter para
/// silenciar los 53 banners del CLI). El backend matmul se resuelve con
/// el mismo criterio auto del CLI (cublas si CUDA, parallel si no).
var resolved_backend: ?@import("matmul").Backend = null;

fn runEngine(st: *ServerState, cli_params: inference.CliParams, sink: TokenSink) !void {
    if (resolved_backend == null) {
        resolved_backend = if (@import("cudaz").isCudaAvailable()) .cublas else .parallel;
    }
    var null_out = quiet.NullWriter.init(&null_buf);
    try inference.runHybridInferenceSink(
        std.Io.Threaded.global_single_threaded.io(),
        st.allocator,
        st.model,
        st.model_path,
        cli_params,
        resolved_backend.?,
        &null_out.interface,
        sink,
    );
}

/// runEngine público para anthropic.zig/ollama.zig (comparten el mismo
/// camino in-process y la misma resolución de backend).
pub fn runEnginePub(st: *ServerState, cli_params: inference.CliParams, sink: TokenSink) !void {
    return runEngine(st, cli_params, sink);
}

var null_buf: [4096]u8 = undefined; // buffer del NullWriter (compartido; serializado por el BatchingLoop)

fn applyStops(sink: *CollectingSink, stop: ?ChatRequest.StopSeq, buf: *[4][]const u8) void {
    switch (stop orelse return) {
        .single => |s| {
            buf[0] = s;
            sink.stop_seqs = buf[0..1];
        },
        .multi => |multi| {
            const n = @min(multi.len, buf.len);
            for (multi[0..n], 0..) |s, i| buf[i] = s;
            sink.stop_seqs = buf[0..n];
        },
    }
}

fn applyStopsSse(sink: *SseSink, stop: ?ChatRequest.StopSeq, buf: *[4][]const u8) void {
    switch (stop orelse return) {
        .single => |s| {
            buf[0] = s;
            sink.stop_seqs = buf[0..1];
        },
        .multi => |multi| {
            const n = @min(multi.len, buf.len);
            for (multi[0..n], 0..) |s, i| buf[i] = s;
            sink.stop_seqs = buf[0..n];
        },
    }
}

// ─── SseSink: TokenSink → chunks SSE OpenAI ──────────────────────────────

pub const SseSink = struct {
    allocator: std.mem.Allocator,
    writer: httpx.StreamWriter,
    model_name: []const u8,
    id: []const u8,
    created: i64,
    /// Stop-sequences: detectadas incrementalmente (corte al vuelo).
    stop_seqs: []const []const u8 = &.{},
    /// Texto acumulado (para detección incremental de stops).
    acc: std.ArrayList(u8) = .empty,
    usage: ?token_sink.Usage = null,
    finish_reason: token_sink.FinishReason = .stop,
    finished: bool = false,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, writer: httpx.StreamWriter, model_name: []const u8) Self {
        const id = std.fmt.allocPrint(allocator, "chatcmpl-{x}", .{rand_u64()}) catch "chatcmpl-err";
        return .{
            .allocator = allocator,
            .writer = writer,
            .model_name = model_name,
            .id = id,
            .created = nowSec(),
        };
    }

    pub fn deinit(self: *Self) void {
        // id: allocPrint o literal; se libera sólo si fue allocado (arena
        // per-request en T2d eliminará esta distinción).
        if (!std.mem.eql(u8, self.id, "chatcmpl-err")) self.allocator.free(self.id);
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
        _ = token; // id del token (no se expone en el chunk SSE)
        if (self.finished) return;

        // Detección incremental de stop-sequences: el texto acumulado
        // termina con algún stop ⇒ cortar AHORA (no emitir más).
        self.acc.appendSlice(self.allocator, text) catch {};
        for (self.stop_seqs) |ss| {
            if (ss.len > 0 and std.mem.endsWith(u8, self.acc.items, ss)) {
                self.finish_reason = .stop;
                return token_sink.SinkSignal.StopSequenceHit;
            }
        }

        // Chunk SSE: chat.completion.chunk con delta.content.
        const payload = jsonStringify(self.allocator, .{
            .id = self.id,
            .object = "chat.completion.chunk",
            .created = self.created,
            .model = self.model_name,
            .choices = .{
                .{
                    .index = 0,
                    .delta = .{ .content = text },
                    .finish_reason = null,
                },
            },
        }) catch return;
        defer self.allocator.free(payload);

        self.writer.writeAll("data: ") catch return; // cliente muerto: drop silencioso
        self.writer.writeAll(payload) catch return;
        self.writer.writeAll("\n\n") catch return;
    }

    fn finishImpl(ctx: *anyopaque, reason: token_sink.FinishReason, usage: token_sink.Usage) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.finished = true;
        self.finish_reason = reason;
        self.usage = usage;
    }

    /// Chunk final: finish_reason (+usage si include_usage).
    pub fn emitFinish(self: *Self, opts: ChatRequest.StreamOptions) !void {
        const finish: []const u8 = switch (self.finish_reason) {
            .length => "length",
            .cancelled => "stop",
            .err => "stop",
            .stop => "stop",
        };
        const payload = try jsonStringify(self.allocator, .{
            .id = self.id,
            .object = "chat.completion.chunk",
            .created = self.created,
            .model = self.model_name,
            .choices = .{
                .{
                    .index = 0,
                    .delta = .{},
                    .finish_reason = finish,
                },
            },
        });
        defer self.allocator.free(payload);
        try self.writer.writeAll("data: ");
        try self.writer.writeAll(payload);
        try self.writer.writeAll("\n\n");

        if (opts.include_usage) {
            const u: token_sink.Usage = self.usage orelse .{ .prompt_tokens = 0, .completion_tokens = 0, .total_ms = 0 };
            const upayload = try jsonStringify(self.allocator, .{
                .id = self.id,
                .object = "chat.completion.chunk",
                .created = self.created,
                .model = self.model_name,
                .choices = &[_]u8{},
                .usage = .{
                    .prompt_tokens = u.prompt_tokens,
                    .completion_tokens = u.completion_tokens,
                    .total_tokens = u.prompt_tokens + u.completion_tokens,
                },
            });
            defer self.allocator.free(upayload);
            try self.writer.writeAll("data: ");
            try self.writer.writeAll(upayload);
            try self.writer.writeAll("\n\n");
        }
    }

    /// Error a mitad de stream: chunk con error.info.
    pub fn emitError(self: *Self, err_name: []const u8) void {
        const payload = jsonStringify(self.allocator, .{
            .id = self.id,
            .object = "chat.completion.chunk",
            .created = self.created,
            .model = self.model_name,
            .choices = &[_]u8{},
            .err_info = .{ .message = err_name, .type = "server_error" },
        }) catch return;
        defer self.allocator.free(payload);
        self.writer.writeAll("data: ") catch return;
        self.writer.writeAll(payload) catch return;
        self.writer.writeAll("\n\n") catch return;
    }
};

// ─── GET /v1/models ───────────────────────────────────────────────────────

pub fn models(ctx: *httpx.Context) !httpx.Response {
    const st = getState();
    return ctx.json(.{
        .object = "list",
        .data = .{
            .{
                .id = st.model_id,
                .object = "model",
                .created = nowSec(),
                .owned_by = "zig-ai",
                .root = st.model_path,
                .max_model_len = st.context_length,
                .context_length = st.context_length,
                .architecture = st.architecture,
                .supported_tools = true,
            },
        },
    });
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn roleFromString(s: []const u8) ?chat_template.Role {
    if (std.mem.eql(u8, s, "system")) return .system;
    if (std.mem.eql(u8, s, "user")) return .user;
    if (std.mem.eql(u8, s, "assistant")) return .assistant;
    if (std.mem.eql(u8, s, "tool")) return .tool;
    return null;
}

// ─── POST /v1/completions (legacy text completion) ────────────────────────

const CompletionReq = struct {
    model: []const u8,
    prompt: []const u8,
    max_tokens: ?usize = null,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    seed: ?u64 = null,
    stream: ?bool = null,
    stop: ?[]const u8 = null,
};

pub fn legacyCompletions(ctx: *httpx.Context) !httpx.Response {
    const st = getState();
    const body = ctx.request.body orelse
        return errorResponse(ctx, 400, "invalid_request_error", "missing body");
    const parsed = std.json.parseFromSlice(CompletionReq, ctx.allocator, body, .{
        .ignore_unknown_fields = true,
    }) catch
        return errorResponse(ctx, 400, "invalid_request_error", "invalid JSON");
    defer parsed.deinit();
    const req = parsed.value;

    validation.modelName(req.model) catch
        return errorResponse(ctx, 400, "invalid_request_error", "model must be non-empty");
    validation.messageContent(req.prompt) catch
        return errorResponse(ctx, 400, "invalid_request_error", "prompt exceeds 1 MiB");
    const temp = validation.temperature(req.temperature) catch
        return errorResponse(ctx, 400, "invalid_request_error", "temperature must be in [0,2]");
    const top_p = validation.topP(req.top_p) catch
        return errorResponse(ctx, 400, "invalid_request_error", "top_p must be in (0,1]");
    const prompt_est = @divFloor(req.prompt.len, 4);
    const max_tokens = validation.maxTokens(req.max_tokens, prompt_est, st.context_length) catch
        return errorResponse(ctx, 400, "invalid_request_error", "max_tokens exceeds context window");

    var cli_params = st.backend;
    cli_params.prompt = req.prompt;
    cli_params.max_new_tokens = max_tokens;
    cli_params.sampler = .{ .temperature = temp, .top_p = top_p };
    cli_params.seed = req.seed orelse 42;

    var sink_ctx = CollectingSink.init(ctx.allocator);
    defer sink_ctx.deinit();
    if (req.stop) |s| {
        var arr = [_][]const u8{s};
        sink_ctx.stop_seqs = arr[0..1];
    }

    runEngine(st, cli_params, sink_ctx.sink()) catch |e| {
        return errorResponse(ctx, 500, "server_error", @errorName(e));
    };

    const usage: token_sink.Usage = sink_ctx.usage orelse .{ .prompt_tokens = 0, .completion_tokens = 0, .total_ms = 0 };
    const id = try std.fmt.allocPrint(ctx.allocator, "cmpl-{x}", .{rand_u64()});
    defer ctx.allocator.free(id);

    return ctx.json(.{
        .id = id,
        .object = "text_completion",
        .created = nowSec(),
        .model = req.model,
        .choices = .{
            .{
                .text = sink_ctx.text.items,
                .index = 0,
                .finish_reason = if (sink_ctx.finish_reason == .length) "length" else "stop",
            },
        },
        .usage = .{
            .prompt_tokens = usage.prompt_tokens,
            .completion_tokens = usage.completion_tokens,
            .total_tokens = usage.prompt_tokens + usage.completion_tokens,
        },
    });
}

// ─── Tests ───────────────────────────────────────────────────────────────

test "roleFromString covers OpenAI roles" {
    try std.testing.expectEqual(chat_template.Role.system, roleFromString("system").?);
    try std.testing.expectEqual(chat_template.Role.user, roleFromString("user").?);
    try std.testing.expectEqual(chat_template.Role.assistant, roleFromString("assistant").?);
    try std.testing.expectEqual(chat_template.Role.tool, roleFromString("tool").?);
    try std.testing.expectEqual(@as(?chat_template.Role, null), roleFromString("bogus"));
}
