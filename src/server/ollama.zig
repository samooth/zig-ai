//! Ollama-compatible API v2: /api/chat, /api/generate, /api/tags, /api/show.
//!
//! T7 del plan server F2. Reemplaza el pseudo-streaming de Fase 1:
//! el engine corre IN-PROCESS (runHybridInferenceSink vía
//! openai.runEnginePub) y cada token emitido se serializa como línea
//! NDJSON al vuelo (Ollama usa NDJSON, no SSE).
//!
//! Schema fiel a https://github.com/ollama/ollama/blob/main/docs/api.md
//! Stream (/api/chat y /api/generate con stream:true):
//!   {"model":"...","created_at":"...","message":{...},"done":false}\n
//!   ...una línea POR TOKEN...
//!   {"model":"...","created_at":"...","done":true,"done_reason":"stop",
//!    "total_duration":ns,"prompt_eval_count":N,"eval_count":M}\n
//!
//! Auth: Ollama no define auth propio; mod.zig aplica secured.wrap
//! (Bearer/x-api-key) — clientes Ollama estándar no envían key, así que
//! el operador elige: loopback sin keys = open, público = keys.

const std = @import("std");
const httpx = @import("httpx");
const chat_template = @import("chat_template");
const inference = @import("inference");
const time_mod = @import("time");
const json_util = @import("json_util");
const token_sink = @import("token_sink");
const validation = @import("validation");

const TokenSink = token_sink.TokenSink;
const CollectingSink = token_sink.CollectingSink;

const jsonStringify = json_util.jsonStringify;
const rand_u64 = json_util.rand_u64;
const Message = chat_template.Message;
const Role = chat_template.Role;

fn nowSec() i64 {
    return time_mod.wallClockSec();
}

// ─── Estado del server (compartido vía mod.zig) ────────────────────────────

const openai = @import("openai");
const ServerState = openai.ServerState;

var g_state: ?*ServerState = null;

pub fn setState(st: *ServerState) void {
    g_state = st;
}

fn getState() *ServerState {
    return g_state orelse unreachable;
}

// ─── RFC3339 timestamp (formato Ollama created_at) ─────────────────────────

/// Ollama usa RFC3339 con nanos: 2026-09-05T12:00:00.000000000Z.
/// Vía std.time.epoch (conversión civil testeada por la stdlib).
/// Nanos fijos a .000000000 (suficiente para clientes Ollama).
fn rfc3339(buf: []u8) []const u8 {
    return rfc3339FromSec(buf, @intCast(@max(nowSec(), 0)));
}

fn rfc3339FromSec(buf: []u8, secs: u64) []const u8 {
    const es = std.time.epoch.EpochSeconds{ .secs = secs };
    const epoch_day = es.getEpochDay();
    const year_day = epoch_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_secs = es.getDaySeconds();

    return std.fmt.bufPrint(buf, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}.000000000Z", .{
        year_day.year,
        @intFromEnum(month_day.month),
        month_day.day_index + 1,
        day_secs.getHoursIntoDay(),
        day_secs.getMinutesIntoHour(),
        day_secs.getSecondsIntoMinute(),
    }) catch "1970-01-01T00:00:00.000000000Z";
}

// ─── Request schemas ───────────────────────────────────────────────────────

const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
};

const Options = struct {
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?usize = null,
    seed: ?u64 = null,
    num_predict: ?usize = null,
    stop: ?[]const u8 = null,
};

const ChatReq = struct {
    model: []const u8,
    messages: []ChatMessage,
    stream: ?bool = null,
    options: ?Options = null,
};

const GenerateReq = struct {
    model: []const u8,
    prompt: ?[]const u8 = null,
    suffix: ?[]const u8 = null,
    system: ?[]const u8 = null,
    template: ?[]const u8 = null,
    context: ?[]u64 = null, // encoded context from previous response
    stream: ?bool = null,
    raw: ?bool = null,
    options: ?Options = null,
};

// ─── Error responses (formato Ollama: {"error":"..."}) ─────────────────────

fn errorResponse(ctx: *httpx.Context, status: u16, msg: []const u8) !httpx.Response {
    const body = try jsonStringify(ctx.allocator, .{
        .err = msg,
    });
    return httpx.Response.fromJson(ctx.allocator, status, body);
}

// ─── POST /api/chat ───────────────────────────────────────────────────────

pub fn chat(ctx: *httpx.Context) !httpx.Response {
    const st = getState();
    const body = ctx.request.body orelse
        return errorResponse(ctx, 400, "missing body");
    const parsed = std.json.parseFromSlice(ChatReq, ctx.allocator, body, .{
        .ignore_unknown_fields = true,
    }) catch
        return errorResponse(ctx, 400, "invalid JSON");
    defer parsed.deinit();
    const req = parsed.value;

    // Validation.
    validation.modelName(req.model) catch
        return errorResponse(ctx, 400, "model must be non-empty");
    validation.messages(req.messages.len) catch
        return errorResponse(ctx, 400, "messages: 1..128 items");
    for (req.messages) |m| validation.messageContent(m.content) catch
        return errorResponse(ctx, 400, "message content exceeds 1 MiB");

    const opts = req.options orelse Options{};
    if (!validateOpts(opts)) return errorResponse(ctx, 400, "invalid options");

    // Messages → chat template (system incluido en el array en Ollama).
    var msgs: std.ArrayList(Message) = .empty;
    defer msgs.deinit(ctx.allocator);
    for (req.messages) |m| {
        const role: Role = if (std.mem.eql(u8, m.role, "system"))
            .system
        else if (std.mem.eql(u8, m.role, "user"))
            .user
        else if (std.mem.eql(u8, m.role, "assistant"))
            .assistant
        else if (std.mem.eql(u8, m.role, "tool"))
            .tool
        else
            return errorResponse(ctx, 400, "invalid role");
        try msgs.append(ctx.allocator, .{ .role = role, .content = m.content });
    }
    const kind = chat_template.detectTemplateKind(st.model_id);
    const prompt = try chat_template.render(ctx.allocator, kind, msgs.items);
    defer ctx.allocator.free(prompt);

    const cli_params = optsToParams(st, opts, prompt, msgs.items.len);
    var stop_buf: [1][]const u8 = undefined;
    var stops: []const []const u8 = &.{};
    if (stopFromOpts(opts)) |s| {
        stop_buf[0] = s;
        stops = stop_buf[0..1];
    }

    if (req.stream orelse false) {
        return streamChat(ctx, st, cli_params, req.model, stops);
    } else {
        return blockChat(ctx, st, cli_params, req.model, stops);
    }
}

// ─── POST /api/generate ────────────────────────────────────────────────────

pub fn generate(ctx: *httpx.Context) !httpx.Response {
    const st = getState();
    const body = ctx.request.body orelse
        return errorResponse(ctx, 400, "missing body");
    const parsed = std.json.parseFromSlice(GenerateReq, ctx.allocator, body, .{
        .ignore_unknown_fields = true,
    }) catch
        return errorResponse(ctx, 400, "invalid JSON");
    defer parsed.deinit();
    const req = parsed.value;

    const opts = req.options orelse Options{};
    if (!validateOpts(opts)) return errorResponse(ctx, 400, "invalid options");

    // raw=true: prompt crudo sin template. raw=false (default): envolver
    // como user-message del template del modelo (comportamiento Ollama).
    var prompt: []const u8 = undefined;
    if (req.raw orelse false) {
        prompt = req.prompt orelse "";
    } else {
        var msgs: std.ArrayList(Message) = .empty;
        defer msgs.deinit(ctx.allocator);
        if (req.system) |sys| {
            try msgs.append(ctx.allocator, .{ .role = .system, .content = sys });
        }
        try msgs.append(ctx.allocator, .{
            .role = .user,
            .content = req.prompt orelse "",
        });
        const kind = chat_template.detectTemplateKind(st.model_id);
        prompt = try chat_template.render(ctx.allocator, kind, msgs.items);
    }
    defer ctx.allocator.free(prompt);

    const cli_params = optsToParams(st, opts, prompt, 1);
    var stop_buf: [1][]const u8 = undefined;
    var stops: []const []const u8 = &.{};
    if (stopFromOpts(opts)) |s| {
        stop_buf[0] = s;
        stops = stop_buf[0..1];
    }

    if (req.stream orelse false) {
        return streamGenerate(ctx, st, cli_params, req.model, stops);
    } else {
        return blockGenerate(ctx, st, cli_params, req.model, stops);
    }
}

// ─── Caminos /api/chat ─────────────────────────────────────────────────────

fn blockChat(
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
        return errorResponse(ctx, 500, @errorName(e));
    };

    const usage: token_sink.Usage = sink_ctx.usage orelse .{
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .total_ms = 0,
    };
    const done_reason: []const u8 = if (sink_ctx.finish_reason == .length) "length" else "stop";
    var ts_buf: [64]u8 = undefined;

    return ctx.json(.{
        .model = model_name,
        .created_at = rfc3339(&ts_buf),
        .message = .{
            .role = "assistant",
            .content = sink_ctx.text.items,
        },
        .done_reason = done_reason,
        .done = true,
        .total_duration = usage.total_ms * 1_000_000,
        .prompt_eval_count = usage.prompt_tokens,
        .eval_count = usage.completion_tokens,
    });
}

fn streamChat(
    ctx: *httpx.Context,
    st: *ServerState,
    cli_params: inference.CliParams,
    model_name: []const u8,
    stops: []const []const u8,
) !httpx.Response {
    const writer = try startNdjson(ctx);
    var sink_ctx = NdjsonSink.init(ctx.allocator, writer, model_name, .chat);
    defer sink_ctx.deinit();
    var stops_arr: [4][]const u8 = undefined;
    copyStops(&stops_arr, stops);
    sink_ctx.stop_seqs = stops_arr[0..stops.len];

    openai.runEnginePub(st, cli_params, sink_ctx.sink()) catch |e| {
        sink_ctx.emitError(e);
        return httpx.Response.fromText(ctx.allocator, 200, "");
    };
    sink_ctx.emitTrailers();
    return httpx.Response.fromText(ctx.allocator, 200, "");
}

// ─── Caminos /api/generate ─────────────────────────────────────────────────

fn blockGenerate(
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
        return errorResponse(ctx, 500, @errorName(e));
    };

    const usage: token_sink.Usage = sink_ctx.usage orelse .{
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .total_ms = 0,
    };
    const done_reason: []const u8 = if (sink_ctx.finish_reason == .length) "length" else "stop";
    var ts_buf: [64]u8 = undefined;

    return ctx.json(.{
        .model = model_name,
        .created_at = rfc3339(&ts_buf),
        .response = sink_ctx.text.items,
        .done = true,
        .done_reason = done_reason,
        .total_duration = usage.total_ms * 1_000_000,
        .prompt_eval_count = usage.prompt_tokens,
        .eval_count = usage.completion_tokens,
    });
}

fn streamGenerate(
    ctx: *httpx.Context,
    st: *ServerState,
    cli_params: inference.CliParams,
    model_name: []const u8,
    stops: []const []const u8,
) !httpx.Response {
    const writer = try startNdjson(ctx);
    var sink_ctx = NdjsonSink.init(ctx.allocator, writer, model_name, .generate);
    defer sink_ctx.deinit();
    var stops_arr: [4][]const u8 = undefined;
    copyStops(&stops_arr, stops);
    sink_ctx.stop_seqs = stops_arr[0..stops.len];

    openai.runEnginePub(st, cli_params, sink_ctx.sink()) catch |e| {
        sink_ctx.emitError(e);
        return httpx.Response.fromText(ctx.allocator, 200, "");
    };
    sink_ctx.emitTrailers();
    return httpx.Response.fromText(ctx.allocator, 200, "");
}

// ─── NDJSON streaming (startRawStreaming + línea por token) ────────────────

/// Inicia respuesta NDJSON en crudo: 200 + Content-Type: application/x-ndjson.
/// NDJSON tiene framing propio (\n por línea) — sin chunked ni SSE.
fn startNdjson(ctx: *httpx.Context) !httpx.StreamWriter {
    var hdrs = httpx.Headers.init(ctx.allocator);
    defer hdrs.deinit();
    try hdrs.append(httpx.HeaderName.CONTENT_TYPE, "application/x-ndjson");
    try hdrs.append(httpx.HeaderName.CACHE_CONTROL, "no-cache");
    return ctx.startRawStreaming(200, &hdrs);
}

fn copyStops(dst: *[4][]const u8, src: []const []const u8) void {
    for (src, 0..) |s, i| dst[i] = s;
}

// ─── NdjsonSink: TokenSink → líneas NDJSON Ollama ─────────────────────────

pub const NdjsonSink = struct {
    /// /api/chat emite "message", /api/generate emite "response".
    const Mode = enum { chat, generate };

    allocator: std.mem.Allocator,
    writer: httpx.StreamWriter,
    model_name: []const u8,
    mode: Mode,
    stop_seqs: []const []const u8 = &.{},
    /// Texto acumulado (detección incremental de stops).
    acc: std.ArrayList(u8) = .empty,
    usage: ?token_sink.Usage = null,
    finish_reason: token_sink.FinishReason = .stop,
    finished: bool = false,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, writer: httpx.StreamWriter, model_name: []const u8, mode: Mode) Self {
        return .{
            .allocator = allocator,
            .writer = writer,
            .model_name = model_name,
            .mode = mode,
        };
    }

    pub fn deinit(self: *Self) void {
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

        // Detección incremental de stop-sequence.
        self.acc.appendSlice(self.allocator, text) catch {};
        if (self.hitStop()) return token_sink.SinkSignal.StopSequenceHit;

        const line = self.jsonLine(text) catch return;
        self.writer.writeAll(line) catch {};
        self.writer.writeAll("\n") catch {};
        self.allocator.free(line);
    }

    fn finishImpl(ctx: *anyopaque, reason: token_sink.FinishReason, usage: token_sink.Usage) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.finish_reason = reason;
        self.usage = usage;
    }

    fn hitStop(self: *Self) bool {
        for (self.stop_seqs) |ss| {
            if (ss.len > 0 and std.mem.endsWith(u8, self.acc.items, ss)) return true;
        }
        return false;
    }

    /// Línea NDJSON por token (done:false).
    fn jsonLine(self: *Self, text: []const u8) ![]u8 {
        var ts_buf: [64]u8 = undefined;
        const created = rfc3339(&ts_buf);
        return switch (self.mode) {
            .chat => jsonStringify(self.allocator, .{
                .model = self.model_name,
                .created_at = created,
                .message = .{ .role = "assistant", .content = text },
                .done = false,
            }),
            .generate => jsonStringify(self.allocator, .{
                .model = self.model_name,
                .created_at = created,
                .response = text,
                .done = false,
            }),
        };
    }

    /// Línea final done:true con métricas.
    pub fn emitTrailers(self: *Self) void {
        var ts_buf: [64]u8 = undefined;
        const created = rfc3339(&ts_buf);
        const usage = self.usage orelse token_sink.Usage{
            .prompt_tokens = 0,
            .completion_tokens = 0,
            .total_ms = 0,
        };
        const done_reason: []const u8 = if (self.finish_reason == .length) "length" else "stop";
        const line = switch (self.mode) {
            .chat => jsonStringify(self.allocator, .{
                .model = self.model_name,
                .created_at = created,
                .done_reason = done_reason,
                .done = true,
                .total_duration = usage.total_ms * 1_000_000,
                .prompt_eval_count = usage.prompt_tokens,
                .eval_count = usage.completion_tokens,
            }),
            .generate => jsonStringify(self.allocator, .{
                .model = self.model_name,
                .created_at = created,
                .done_reason = done_reason,
                .done = true,
                .total_duration = usage.total_ms * 1_000_000,
                .prompt_eval_count = usage.prompt_tokens,
                .eval_count = usage.completion_tokens,
            }),
        } catch return;
        defer self.allocator.free(line);
        self.writer.writeAll(line) catch {};
        self.writer.writeAll("\n") catch {};
    }

    /// Error mid-stream: línea {"error":"..."} + done:true.
    pub fn emitError(self: *Self, e: anyerror) void {
        const line = jsonStringify(self.allocator, .{
            .model = self.model_name,
            .err = @errorName(e),
            .done = true,
        }) catch return;
        defer self.allocator.free(line);
        self.writer.writeAll(line) catch {};
        self.writer.writeAll("\n") catch {};
    }
};

// ─── Helpers options → cli_params ─────────────────────────────────────────

fn validateOpts(opts: Options) bool {
    const ok_temp = validation.temperature(opts.temperature) catch return false;
    _ = ok_temp;
    _ = validation.topP(opts.top_p) catch return false;
    _ = validation.topK(opts.top_k) catch return false;
    if (opts.stop) |s| {
        const one = [_][]const u8{s};
        _ = validation.stopSequences(&one) catch return false;
    }
    return true;
}

/// Options Ollama → CliParams del engine (defaults Ollama: 0.8/0.9/40).
fn optsToParams(st: *ServerState, opts: Options, prompt: []const u8, n_msgs: usize) inference.CliParams {
    _ = n_msgs;
    var p = st.backend;
    p.prompt = prompt;
    p.max_new_tokens = opts.num_predict orelse 256;
    p.sampler = .{
        .temperature = opts.temperature orelse 0.8,
        .top_p = opts.top_p orelse 0.9,
        .top_k = opts.top_k orelse 40,
    };
    p.seed = opts.seed orelse 42;
    return p;
}

/// Stop de Options Ollama (single string) → slice estático del caller.
/// Retorna null si no hay stop (el caller mantiene el buffer vivo).
fn stopFromOpts(opts: Options) ?[]const u8 {
    return opts.stop;
}

// ─── GET /api/tags ────────────────────────────────────────────────────────

pub fn tags(ctx: *httpx.Context) !httpx.Response {
    const st = getState();
    var ts_buf: [64]u8 = undefined;
    const model_size: u64 = blk: {
        const s = std.Io.Dir.cwd().statFile(
            std.Io.Threaded.global_single_threaded.io(),
            st.model_path,
            .{},
        ) catch break :blk 0;
        break :blk s.size;
    };
    return ctx.json(.{
        .models = .{
            .{
                .name = st.model_id,
                .model = st.model_id,
                .modified_at = rfc3339(&ts_buf),
                .size = model_size,
                .digest = "sha256:zig-ai",
                .details = .{
                    .format = "gguf",
                    .family = st.architecture,
                    .parameter_size = "unknown",
                    .quantization_level = "unknown",
                    .context_length = st.context_length,
                },
            },
        },
    });
}

// ─── POST /api/show ───────────────────────────────────────────────────────

pub fn show(ctx: *httpx.Context) !httpx.Response {
    const st = getState();
    const body = ctx.request.body orelse
        return errorResponse(ctx, 400, "missing body");
    const ShowReq = struct { model: ?[]const u8 = null };
    const parsed = std.json.parseFromSlice(ShowReq, ctx.allocator, body, .{
        .ignore_unknown_fields = true,
    }) catch
        return errorResponse(ctx, 400, "invalid JSON");
    defer parsed.deinit();

    var modelfile_buf: [512]u8 = undefined;
    const modelfile = std.fmt.bufPrint(
        &modelfile_buf,
        "# Modelfile generated by zig-ai\nFROM {s}\n",
        .{st.model_path},
    ) catch "# Modelfile\n";
    return ctx.json(.{
        .modelfile = modelfile,
        .parameters = "stop \"<|im_end|>\"",
        .template = "{{ .Prompt }}",
        .details = .{
            .format = "gguf",
            .family = st.architecture,
            .parameter_size = "unknown",
            .quantization_level = "unknown",
            .context_length = st.context_length,
        },
    });
}

test "ollama rfc3339" {
    var buf: [64]u8 = undefined;
    // Valor conocido: 1622924906 = 2021-06-05T20:28:26Z (test de std.time.epoch)
    const ts = rfc3339FromSec(&buf, 1622924906);
    try std.testing.expectEqualStrings("2021-06-05T20:28:26.000000000Z", ts);
    // Formato completo: 34 chars
    try std.testing.expectEqual(@as(usize, 34), ts.len);
    const ts0 = rfc3339FromSec(&buf, 0);
    try std.testing.expectEqualStrings("1970-01-01T00:00:00.000000000Z", ts0);
}

test "ollama copyStops" {
    var arr: [4][]const u8 = undefined;
    const stops = [_][]const u8{"x"};
    copyStops(&arr, &stops);
    try std.testing.expectEqualStrings("x", arr[0]);
}
