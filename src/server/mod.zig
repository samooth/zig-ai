//! Servidor HTTP compatible con OpenAI / Anthropic / Ollama.
//!
//! v2 (server F2): engine IN-PROCESS — el modelo se carga UNA vez en
//! `runServer` y los handlers OpenAI (T5) lo usan vía `ServerState`
//! (openai.setState). Anthropic/Ollama siguen en Fase 1 (Engine
//! sub-proceso) hasta sus REWRITE (T6/T7).
//!
//! Seguridad (T3):
//!   - Auth API keys sha256 + timing-safe (ZIG_AI_API_KEY o file 0600)
//!   - Rate limit por IP (httpx middleware)
//!   - Audit JSON-lines fail-soft
//!
//! Entry point: `runServer(allocator, config) !void` desde main.zig `--serve`.

const std = @import("std");
const httpx = @import("httpx");
const openai = @import("openai");
const anthropic = @import("anthropic");
const ollama = @import("ollama");
const engine = @import("engine");
const auth_mod = @import("auth");
const audit_mod = @import("audit");
const token_sink_mod = @import("token_sink");
const gguf_model = @import("gguf_model");
const inference = @import("inference");
const time_mod = @import("time");
const debugz = @import("debug");

pub const Engine = engine.Engine;

pub const ServerConfig = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    /// Path al modelo GGUF (UN modelo por server — decisión de diseño).
    model_path: []const u8,
    /// CliParams base (ctx, cuantización KV, capas GPU) heredados del CLI.
    cli_params: inference.CliParams,
    /// API keys: file con una key por línea (permisos 0600 verificada al
    /// leer) o env ZIG_AI_API_KEY. Vacío = auth disabled (loopback only).
    api_key_file: ?[]const u8 = null,
    /// Path del audit log JSON-lines (default: no audit).
    audit_log: ?[]const u8 = null,
    /// Rate limit por IP (default 60 req/min).
    rate_limit_rpm: u32 = 60,
    /// TLS: cert+key PEM (httpx TLS nativo). Política: bind no-loopback
    /// SIN TLS es rechazado con error fatal en arranque.
    tls_cert: ?[]const u8 = null,
    tls_key: ?[]const u8 = null,
    /// Warmup: 1 inferencia corta al arranque (loading→ready en /health).
    warmup: bool = true,
    /// Si true, no se bindea al puerto — tests de setup.
    dry_run: bool = false,
};

// ─── Estado global del server ─────────────────────────────────────────────

/// Estado de salud reportado por /health durante y tras el arranque.
pub const HealthState = enum {
    loading, // modelo cargando / warmup corriendo
    ready, // todo montado, sirviendo
    failed, // error fatal en arranque (warmup KO)
};

var g_state: ?*openai.ServerState = null;
var g_auth: ?*auth_mod.Auth = null;
var g_audit: ?*audit_mod.Audit = null;
var g_health: HealthState = .loading;

/// Fail-soft: si el warmup falla, el server queda en failed (503 en
/// /health) pero sigue vivo para diagnóstico.
var g_warmup_error: ?[]const u8 = null;

pub fn healthState() HealthState {
    return g_health;
}

fn getState() *openai.ServerState {
    return g_state orelse unreachable;
}

/// Auth/audit helpers usados por el wrapper de handlers.
fn authEnabled() bool {
    return g_auth != null;
}

/// Wrapper de handlers con auth + audit (aplicado a las rutas de inferencia).
const secured = struct {
    fn wrap(
        comptime handler: *const fn (*httpx.Context) anyerror!httpx.Response,
    ) *const fn (*httpx.Context) anyerror!httpx.Response {
        return struct {
            fn handle(ctx: *httpx.Context) anyerror!httpx.Response {
                const t0 = time_mod.Timer.now();
                var status: u16 = 200;

                // Auth (timing-safe) — 401 sin revelar por qué.
                if (authEnabled()) {
                    const token = auth_mod.extractToken(
                        ctx.header("Authorization"),
                        ctx.header("x-api-key"),
                    ) orelse {
                        status = 401;
                        logAudit(ctx, status, t0, 0);
                        return httpx.Response.fromText(ctx.allocator, 401, "Unauthorized");
                    };
                    if (g_auth.?.validate(token) == null) {
                        status = 401;
                        logAudit(ctx, status, t0, 0);
                        return httpx.Response.fromText(ctx.allocator, 401, "Unauthorized");
                    }
                }

                const resp = handler(ctx) catch |e| {
                    status = 500;
                    logAudit(ctx, status, t0, 0);
                    return e;
                };
                status = resp.status.code;
                logAudit(ctx, status, t0, 0);
                return resp;
            }
        }.handle;
    }

    fn logAudit(ctx: *httpx.Context, status: u16, t0: i128, tokens: usize) void {
        const audit = g_audit orelse return;
        _ = tokens;
        audit.log(.{
            .key_id = null,
            .ip = ctx.connectionIp(),
            .method = @tagName(ctx.request.method),
            .path = ctx.request.uri.path,
            .status = status,
            .latency_ms = @intCast(@divTrunc(time_mod.Timer.now() - t0, std.time.ns_per_ms)),
        });
    }
};

// ─── Arranque ─────────────────────────────────────────────────────────────

pub fn runServer(allocator: std.mem.Allocator, config: ServerConfig) !void {
    // 1. Cargar modelo UNA vez (in-process).
    debugz.dbg.print("[server] cargando modelo: {s}\n", .{config.model_path});
    var model = gguf_model.GgufModel.load(
        std.Io.Threaded.global_single_threaded.io(),
        allocator,
        config.model_path,
    ) catch |err| {
        debugz.dbg.print("[server] FATAL cargando modelo: {any}\n", .{err});
        return err;
    };
    defer model.deinit();

    const model_id = std.fs.path.basename(config.model_path);
    var state = openai.ServerState{
        .allocator = allocator,
        .model = &model,
        .model_path = config.model_path,
        .backend = config.cli_params,
        .model_id = model_id,
        .context_length = @intCast(model.config.context_length),
        .architecture = model.config.architecture,
    };
    openai.setState(&state);
    g_state = &state;

    // 2. Auth (opcional; ZIG_AI_API_KEY o file 0600). Loopback sin keys
    //    está permitido; bind público sin keys es rechazado por Auth.load.
    var auth = auth_mod.Auth.init(allocator);
    defer auth.deinit();
    const is_loopback = std.mem.eql(u8, config.host, "127.0.0.1") or
        std.mem.eql(u8, config.host, "localhost") or
        std.mem.eql(u8, config.host, "::1");
    const env_key: ?[]const u8 = if (std.c.getenv("ZIG_AI_API_KEY")) |k| std.mem.span(k) else null;
    auth.load(config.api_key_file, env_key, is_loopback) catch |err| {
        debugz.dbg.print("[server] FATAL auth: {any}\n", .{err});
        return err;
    };
    if (auth.enabled()) {
        g_auth = &auth;
        debugz.dbg.print("[server] auth: enabled ({d} keys)\n", .{auth.keyCount()});
    } else {
        debugz.dbg.print("[server] auth: disabled (loopback only)\n", .{});
    }

    // 3. Audit log (fail-soft).
    var audit = audit_mod.Audit.init(allocator, .{
        .path = config.audit_log orelse "/tmp/zig-ai-audit.jsonl",
    });
    defer audit.close();
    audit.open() catch |err| {
        debugz.dbg.print("[server] audit disabled ({any}); continuando\n", .{err});
    };
    g_audit = &audit;

    if (config.dry_run) {
        debugz.dbg.print("[server] dry_run=true, no se bindea puerto\n", .{});
        return;
    }

    // 4. Política TLS: bind no-loopback SIN TLS = error fatal (no servimos
    //    plaintext en red pública). Loopback sin TLS permitido (dev local).
    const is_loopback2 = std.mem.eql(u8, config.host, "127.0.0.1") or
        std.mem.eql(u8, config.host, "localhost") or
        std.mem.eql(u8, config.host, "::1");
    const have_tls = config.tls_cert != null and config.tls_key != null;
    if (!is_loopback2 and !have_tls) {
        debugz.dbg.print(
            "[server] FATAL: bind a {s} (no-loopback) sin --tls-cert/--tls-key — rechazado. Usa TLS o --host 127.0.0.1\n",
            .{config.host},
        );
        return error.TlsRequiredForPublicBind;
    }
    if ((config.tls_cert != null) != (config.tls_key != null)) {
        debugz.dbg.print("[server] FATAL: --tls-cert y --tls-key van juntos\n", .{});
        return error.TlsConfigIncomplete;
    }

    // 5. Warmup: 1 inferencia corta ANTES de escuchar — el primer request
    //    de usuario no paga el cold-start (carga capas, JIT graphs, caches).
    //    Fail-soft: si falla, health queda failed (503) pero el server vive.
    if (config.warmup) {
        debugz.dbg.print("[server] warmup: inferencia inicial...\n", .{});
        var warm_params = config.cli_params;
        warm_params.prompt = "Hi";
        warm_params.max_new_tokens = 1;
        var warm_sink = token_sink_mod.CollectingSink.init(allocator);
        defer warm_sink.deinit();
        openai.runEnginePub(&state, warm_params, warm_sink.sink()) catch |err| {
            g_health = .failed;
            g_warmup_error = @errorName(err);
            debugz.dbg.print("[server] warmup FALLÓ: {s} — health=failed (503)\n", .{@errorName(err)});
        };
        if (g_health != .failed) {
            debugz.dbg.print("[server] warmup OK ({d} tokens) — health=ready\n", .{
                warm_sink.usage.?.completion_tokens,
            });
        }
    } else {
        debugz.dbg.print("[server] warmup desactivado — health=ready\n", .{});
    }
    if (g_health != .failed) g_health = .ready;

    // 6. HTTP server.
    var server = httpx.Server.initWithConfig(allocator, .{
        .host = config.host,
        .port = config.port,
        .port_conflict = .fail,
        .max_connections = 256,
        .keep_alive = true,
        .tls_enabled = have_tls,
        .tls_cert_path = config.tls_cert,
        .tls_key_path = config.tls_key,
    });
    defer server.deinit();

    // Rate limit por IP (connection socket, no headers — anti-spoof).
    // httpx rateLimit toma la config comptime; el valor runtime va por
    // ServerConfig pero el middleware se fija a 60/min v1 (TODO T8: comptime
    // switch por rangos 30/60/120).
    _ = config.rate_limit_rpm;
    try server.use(httpx.middleware.rateLimit(.{
        .max_requests = 60,
        .window_ms = 60_000,
        .trust_proxy_headers = false,
    }));

    // ─── Health & root ─────────────────────────────────────────────────
    try server.get("/health", healthHandler());
    try server.get("/", rootHandler());

    // ─── OpenAI (T5 — engine in-process, SSE real) ─────────────────────
    try server.post("/v1/chat/completions", secured.wrap(openai.chatCompletions));
    try server.post("/v1/completions", secured.wrap(openai.legacyCompletions));
    try server.get("/v1/models", secured.wrap(openai.models));

    // ─── Anthropic (T6 — engine in-process, eventos SSE Anthropic) ────
    anthropic.setState(&state);
    try server.post("/v1/messages", secured.wrap(anthropic.messages));
    try server.post("/v1/messages/count_tokens", secured.wrap(anthropic.countTokens));

    // ─── Ollama (T7 — engine in-process, NDJSON real) ──────────────────
    ollama.setState(&state);
    try server.post("/api/chat", secured.wrap(ollama.chat));
    try server.post("/api/generate", secured.wrap(ollama.generate));
    try server.get("/api/tags", secured.wrap(ollama.tags));
    try server.post("/api/show", secured.wrap(ollama.show));

    debugz.dbg.print("[server] escuchando en http://{s}:{d}\n", .{ config.host, config.port });
    debugz.dbg.print("[server] model={s} arch={s} ctx={d}\n", .{
        state.model_id, state.architecture, state.context_length,
    });

    g_http_server = &server;
    errdefer g_http_server = null;

    try server.listen();
    g_http_server = null;
}

/// Handle al httpx server vivo (para stopTestServer / shutdown externo).
var g_http_server: ?*httpx.Server = null;

/// Detiene el server (tests E2E). No-op si no está corriendo.
pub fn stopTestServer() void {
    if (g_http_server) |s| s.stop();
}

// ─── Health & root handlers ──────────────────────────────────────────────

fn healthHandler() *const fn (*httpx.Context) anyerror!httpx.Response {
    return struct {
        fn handle(_: *httpx.Context) anyerror!httpx.Response {
            const st = getState();
            var buf: [320]u8 = undefined;
            const body = switch (g_health) {
                .ready => std.fmt.bufPrint(
                    &buf,
                    "{{\"status\":\"ok\",\"health\":\"ready\",\"engine\":\"zig-ai\",\"model\":\"{s}\",\"context_length\":{d}}}",
                    .{ st.model_id, st.context_length },
                ) catch unreachable,
                .loading => std.fmt.bufPrint(
                    &buf,
                    "{{\"status\":\"loading\",\"health\":\"loading\",\"engine\":\"zig-ai\"}}",
                    .{},
                ) catch unreachable,
                .failed => blk: {
                    // 503 con el motivo (sin paths internos).
                    var ebuf: [128]u8 = undefined;
                    const emsg = std.fmt.bufPrint(&ebuf, "{s}", .{g_warmup_error orelse "unknown"}) catch "unknown";
                    break :blk std.fmt.bufPrint(
                        &buf,
                        "{{\"status\":\"error\",\"health\":\"failed\",\"engine\":\"zig-ai\",\"error\":\"{s}\"}}",
                        .{emsg},
                    ) catch unreachable;
                },
            };
            const code: u16 = if (g_health == .failed) 503 else 200;
            return httpx.Response.fromText(ctx_alloc(), code, body);
        }
    }.handle;
}

fn rootHandler() *const fn (*httpx.Context) anyerror!httpx.Response {
    return struct {
        fn handle(_: *httpx.Context) anyerror!httpx.Response {
            return httpx.Response.fromText(ctx_alloc(), 200,
                \\zig-ai-engine server
                \\
                \\Endpoints:
                \\  POST /v1/chat/completions      (OpenAI compatible)
                \\  POST /v1/completions            (OpenAI legacy)
                \\  GET  /v1/models
                \\  POST /v1/messages               (Anthropic)
                \\  POST /v1/messages/count_tokens
                \\  POST /api/chat                  (Ollama)
                \\  POST /api/generate              (Ollama)
                \\  GET  /api/tags                  (Ollama)
                \\  POST /api/show                  (Ollama)
                \\
            );
        }
    }.handle;
}

/// Allocator para handlers estáticos (health/root): usa el del ServerState.
/// Los handlers con ctx usan ctx.allocator; estos no reciben state por ctx
/// así que tomamos el global del server (lifetime = runServer).
fn ctx_alloc() std.mem.Allocator {
    return g_state.?.allocator;
}
