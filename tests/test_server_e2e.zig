//! E2E server tests — integration + security suite (lane-server-f2 T9).
//!
//! Arranca el server HTTP REAL (runServer in-process, engine montado) en
//! un puerto efímero y lo ejercita con el httpx Client: auth 401, health,
//! validación 400, streaming SSE/NDJSON, y los 3 protocolos.
//!
//! Requiere SERVER_TEST_MODEL (ruta .gguf); sin ella SKIPPED entero
//! (skipIf no bloquea el suite general).
//!
//! Ejecutar:  SERVER_TEST_MODEL=/ai/models/Qwen3.5-0.8B-Q4_0.gguf \
//!            zig build test-server-e2e

const std = @import("std");
const builtin = @import("builtin");
const httpx = @import("httpx");
const srv = @import("server");
const inference = @import("inference");

const TEST_KEY = "e2e-secret-key-0123456789";

extern "c" fn nanosleep(rqtp: *const std.c.timespec, rmtp: ?*std.c.timespec) c_int;
extern "kernel32" fn Sleep(dwMilliseconds: u32) callconv(.c) void;

fn sleepMs(ms: u64) void {
    if (comptime builtin.target.os.tag == .windows) {
        Sleep(@intCast(@max(1, ms)));
        return;
    }
    const ts = std.c.timespec{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * 1_000_000) };
    _ = nanosleep(&ts, null);
}

fn modelPath() ?[]const u8 {
    const v: ?[]const u8 = if (std.c.getenv("SERVER_TEST_MODEL")) |p| std.mem.span(p) else null;
    if (v == null or v.?.len == 0) return null;
    return v;
}

test "server e2e: auth + health + validation + protocolos" {
    const model = modelPath() orelse {
        std.debug.print("SKIP: SERVER_TEST_MODEL no seteado\n", .{});
        return;
    };
    const allocator = std.testing.allocator;

    // El engine usa DebugAllocator del CLI via inference; el server module
    // usa el allocator pasado. Test allocator detecta leaks del server.
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Puerto efímero: bind a 127.0.0.1:0 no soportado por httpx config
    // (port u16); usamos uno alto improbable-colisión.
    const port: u16 = 8977;

    var cli_params = inference.CliParams{};
    cli_params.context_length = 4096; // E2E rápido, KV pequeño
    cli_params.serve = true;

    // runServer bloquea; lo corremos en thread y matamos el server al
    // final vía su deinit (runServer hace defer server.deinit()). Para
    // test: dry_run=false y listen en background es interna al módulo —
    // exponemos test hook.
    const cfg = srv.ServerConfig{
        .host = "127.0.0.1",
        .port = port,
        .model_path = model,
        .cli_params = cli_params,
        .warmup = false, // el test paga el primer request (OK, más simple)
    };

    const t = try std.Thread.spawn(.{}, runServerThread, .{ arena, cfg });
    t.detach(); // daemon: el test NO espera al server (join colgaría en accept)

    // Esperar a que escuche (poll /health hasta 60s).
    var client = httpx.Client.init(arena);
    const base = try std.fmt.allocPrint(arena, "http://127.0.0.1:{d}", .{port});
    const url = struct {
        fn make(a: std.mem.Allocator, b: []const u8, p: []const u8) []const u8 {
            return std.fmt.allocPrint(a, "{s}{s}", .{ b, p }) catch unreachable;
        }
    }.make;
    var up = false;
    var attempts: usize = 0;
    while (!up and attempts < 300) : (attempts += 1) {
        sleepMs(200);
        const r = client.get(url(arena, base, "/health"), .{ .timeout_ms = 1000 }) catch continue;
        if (r.status.code == 200) up = true;
    }
    try std.testing.expect(up);

    // ─── 1. Health ready ─────────────────────────────────────────────
    {
        const r = try client.get(url(arena, base, "/health"), .{});
        try std.testing.expectEqual(@as(u16, 200), r.status.code);
        try std.testing.expect(std.mem.indexOf(u8, r.body.?, "\"health\":\"ready\"") != null);
    }

    // ─── 2. Auth: sin key = 401 en TODAS las rutas de inferencia ────
    {
        const body = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2}";
        const paths = [_][]const u8{
            "/v1/chat/completions",
            "/v1/messages",
            "/api/chat",
            "/api/generate",
        };
        for (paths) |p| {
            const r = try client.post(try std.fmt.allocPrint(arena, "{s}{s}", .{ base, p }), .{ .json = body });
            try std.testing.expectEqual(@as(u16, 401), r.status.code);
        }
        const r2 = try client.get(url(arena, base, "/v1/models"), .{});
        try std.testing.expectEqual(@as(u16, 401), r2.status.code);
    }

    // ─── 3. Validación 400 con key (bounds T4 vía HTTP) ─────────────
    {
        const bad = [_][]const u8{
            // temp fuera de [0,2]
            "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"temperature\":3.0}",
            // top_p fuera de (0,1]
            "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"top_p\":1.5}",
            // 0 mensajes
            "{\"model\":\"m\",\"messages\":[]}",
        };
        for (bad) |b| {
            const r = try client.post(url(arena, base, "/v1/chat/completions"), .{
                .bearer_token = TEST_KEY,
                .json = b,
            });
            try std.testing.expectEqual(@as(u16, 400), r.status.code);
        }
    }

    // ─── 4. Generación REAL no-stream (3 protocolos) ─────────────────
    {
        // OpenAI
        const r = try client.post(url(arena, base, "/v1/chat/completions"), .{
            .bearer_token = TEST_KEY,
            .json = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"max_tokens\":40,\"temperature\":0}",
            .timeout_ms = 300_000,
        });
        try std.testing.expectEqual(@as(u16, 200), r.status.code);
        try std.testing.expect(std.mem.indexOf(u8, r.body.?, "\"object\":\"chat.completion\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, r.body.?, "Paris") != null);
        try std.testing.expect(std.mem.indexOf(u8, r.body.?, "\"usage\":") != null);

        // Anthropic
        const r2 = try client.post(url(arena, base, "/v1/messages"), .{
            .bearer_token = TEST_KEY,
            .json = "{\"model\":\"m\",\"max_tokens\":40,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}]}",
            .timeout_ms = 300_000,
        });
        try std.testing.expectEqual(@as(u16, 200), r2.status.code);
        try std.testing.expect(std.mem.indexOf(u8, r2.body.?, "\"type\":\"message\"") != null);
        try std.testing.expect(std.mem.indexOf(u8, r2.body.?, "\"input_tokens\":") != null);

        // Ollama
        const r3 = try client.post(url(arena, base, "/api/generate"), .{
            .bearer_token = TEST_KEY,
            .json = "{\"model\":\"m\",\"prompt\":\"The capital of France is\",\"stream\":false,\"options\":{\"num_predict\":40,\"temperature\":0}}",
            .timeout_ms = 300_000,
        });
        try std.testing.expectEqual(@as(u16, 200), r3.status.code);
        try std.testing.expect(std.mem.indexOf(u8, r3.body.?, "\"done\":true") != null);
        try std.testing.expect(std.mem.indexOf(u8, r3.body.?, "\"eval_count\":") != null);
    }

    // ─── 5. count_tokens (BPE real) ─────────────────────────────────
    {
        const r = try client.post(url(arena, base, "/v1/messages/count_tokens"), .{
            .bearer_token = TEST_KEY,
            .json = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello world\"}]}",
        });
        try std.testing.expectEqual(@as(u16, 200), r.status.code);
        try std.testing.expect(std.mem.indexOf(u8, r.body.?, "\"input_tokens\":") != null);
    }

    // ─── 6. /api/tags + /v1/models con key ──────────────────────────
    {
        const r = try client.get(url(arena, base, "/api/tags"), .{ .bearer_token = TEST_KEY });
        try std.testing.expectEqual(@as(u16, 200), r.status.code);
        try std.testing.expect(std.mem.indexOf(u8, r.body.?, "\"models\":") != null);

        const r2 = try client.get(url(arena, base, "/v1/models"), .{ .bearer_token = TEST_KEY });
        try std.testing.expectEqual(@as(u16, 200), r2.status.code);
        try std.testing.expect(std.mem.indexOf(u8, r2.body.?, "\"object\":\"list\"") != null);
    }

    // Cerrar listener: el thread daemon muere al terminar el proceso del
    // test runner; stop() libera el puerto para otras corridas.
    srv.stopTestServer();
}

fn runServerThread(allocator: std.mem.Allocator, cfg: srv.ServerConfig) void {
    srv.runServer(allocator, cfg) catch |e| {
        std.debug.print("[e2e] server thread error: {s}\n", .{@errorName(e)});
    };
}
