//! Engine — wrapper alrededor del binario zig-ai-engine para servirlo por HTTP.
//!
//! Estrategia Fase 1: el `Engine` mantiene un handle al binario y por cada
//! request lanza un sub-proceso `zig-ai-engine` con --prompt y captura stdout.
//! NO es streaming; cada request ejecuta un sub-proceso (cold-start ~1-3s).
//!
//! Tradeoffs explícitos:
//!   - Latencia: ~50ms extra por spawn + (re-)carga del modelo
//!     (modelo re-leído por sub-proceso, no cacheado). Para latencia baja,
//!     Fase 2 refactorizará `runHybridInference` para ser reusable in-process.
//!   - Memoria: cada proceso ocupa la RAM completa del modelo. Se mitiga con
//!     `num_parallel: 1` por defecto y un mutex (--max-requests 1 en v1).
//!   - Correctitud: garantizada por el camino CLI ya validado por tests E2E.
//!
//! Ventajas de Fase 1:
//!   - No toca main.zig (sin refactor masivo).
//!   - Endpoints OpenAI/Anthropic/Ollama ya quedan utilizables para testing.
//!   - Tests del server no requieren modelo real (mockeamos Engine).
//!
//! Fase 2 (TODO): extraer un `HybridInferenceContext` reutilizable in-process
//! que mantenga el modelo cargado y permita streaming token-por-token vía
//! callback. Eso habilitará SSE real en los handlers.

const std = @import("std");
const debugz = @import("debug");
const gguf_model = @import("gguf_model");
const gguf_tokenizer = @import("gguf_tokenizer");
const time = @import("time");

pub const Tokenizer = gguf_tokenizer.GgufTokenizer;

/// Engine global accesible por los handlers. httpx.zig Fase 1 no soporta
/// state per-request vía closure; lo seteamos en `mod.runServer` antes de
/// `server.listen()`. Fase 2 lo reemplaza con middleware nativo.
var g_engine: ?*Engine = null;

pub fn setGlobalEngine(eng: *Engine) void {
    g_engine = eng;
}

pub fn getGlobalEngine() *Engine {
    return g_engine orelse unreachable;
}

pub const CompletionParams = struct {
    prompt: []const u8,
    max_new_tokens: usize = 256,
    temperature: f32 = 0.7,
    top_k: usize = 40,
    top_p: f32 = 0.95,
    seed: u64 = 42,
    /// Si true, también imprimir logits stats (DUMP_LOGITS). No usado en Fase 1.
    dump_logits: bool = false,
};

pub const CompletionResult = struct {
    text: []u8, // owned, allocator
    prompt_tokens: usize,
    completion_tokens: usize,
    total_ms: u64,
    tokens_per_second: f32,

    pub fn deinit(self: CompletionResult, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
    }
};

/// Estado del motor: la ruta al binario + path al modelo + tokenizer cargado.
/// `tokenizer` se carga eagerly (es barato: ~50-200MB en RAM) y se usa para
/// contar tokens del prompt y posiblemente post-procesar.
/// El binario se invoca en `generate`.
pub const Engine = struct {
    allocator: std.mem.Allocator,
    /// Path absoluto al binario zig-ai-engine.
    engine_bin: []const u8,
    /// Path absoluto al modelo GGUF.
    model_path: []const u8,
    /// Tokenizer cargado del GGUF (para token-counting y stop-tokens).
    tokenizer: Tokenizer,
    /// Nombre del modelo (basename del path) para /v1/models.
    model_id: []const u8,
    /// Context length (leído del GGUF).
    context_length: u32,
    /// Architecture (leída del GGUF).
    architecture: []const u8,
    /// BOS token id (de Tokenizer).
    bos_token: ?u32,
    /// EOS token id (de Tokenizer).
    eos_token: ?u32,
    /// Mutex para serializar requests (Fase 1: 1 proceso a la vez).
    /// v2 lo reemplaza por un pool de procesos o in-process reuse.
    /// Zig 0.16 no tiene std.Thread.Mutex, usamos un spinlock casero.
    lock: std.atomic.Value(u32) = .{ .raw = 0 },

    const Self = @This();

    pub const InitError = error{
        EngineBinNotFound,
        ModelLoadFailed,
        TokenizerLoadFailed,
        OutOfMemory,
    };

    /// Carga modelo + tokenizer. NO carga las capas pesadas (eso lo hace el
    /// sub-proceso al ejecutarse); sólo el header GGUF + tokenizer.
    pub fn init(
        allocator: std.mem.Allocator,
        engine_bin: []const u8,
        model_path: []const u8,
    ) InitError!Self {
        // 1) Verificar que el binario existe
        const io = std.Io.Threaded.global_single_threaded.io();
        std.Io.Dir.access(std.Io.Dir.cwd(), io, engine_bin, .{}) catch {
            return InitError.EngineBinNotFound;
        };

        // 2) Cargar modelo (header only) + tokenizer
        var model = gguf_model.GgufModel.load(io, allocator, model_path) catch {
            return InitError.ModelLoadFailed;
        };
        defer model.deinit();

        const cfg = model.config;
        const tok = gguf_tokenizer.GgufTokenizer.fromGguf(allocator, &model.file) catch {
            return InitError.TokenizerLoadFailed;
        };

        // Derivar model_id del basename del path
        const basename = std.fs.path.basename(model_path);

        return .{
            .allocator = allocator,
            .engine_bin = engine_bin,
            .model_path = model_path,
            .tokenizer = tok,
            .model_id = try allocator.dupe(u8, basename),
            .context_length = @intCast(cfg.context_length),
            .architecture = try allocator.dupe(u8, cfg.architecture),
            .bos_token = tok.bos_id,
            .eos_token = tok.eos_id,
            .lock = .{ .raw = 0 },
        };
    }

    pub fn deinit(self: *Self) void {
        self.tokenizer.deinit();
        self.allocator.free(self.model_id);
        self.allocator.free(self.architecture);
    }

    /// Cuenta cuántos tokens ocupa un prompt vía heurística de chars/4.
    /// El tokenizer BPE no está embebido en GgufTokenizer (sólo la tabla de
    /// vocab). Para conteo exacto, Fase 2 cargará BPETokenizer.fromGguf.
    /// Para nuestros propósitos (reporting en `usage`), la heurística es OK.
    pub fn countPromptTokens(self: *Self, text: []const u8) usize {
        _ = self;
        return @divFloor(text.len, 4);
    }

    /// Genera una completion delegando a un sub-proceso zig-ai-engine.
    /// Synchronous. No streaming.
    pub fn generate(self: *Self, params: CompletionParams) !CompletionResult {
        // Spinlock: Fase 1 = 1 request a la vez (modelo completo en RAM por proceso)
        acquireLock(&self.lock);
        defer releaseLock(&self.lock);

        const start = time.Timer.now();

        // Construir args del sub-proceso
        var args_buf: std.ArrayList([]const u8) = .empty;
        defer args_buf.deinit(self.allocator);
        try args_buf.append(self.allocator, self.engine_bin);
        try args_buf.append(self.allocator, "--model");
        try args_buf.append(self.allocator, self.model_path);
        try args_buf.append(self.allocator, "--prompt");
        try args_buf.append(self.allocator, params.prompt);
        var n_str_buf: [32]u8 = undefined;
        const n_str = try std.fmt.bufPrint(&n_str_buf, "{d}", .{params.max_new_tokens});
        try args_buf.append(self.allocator, "-n");
        try args_buf.append(self.allocator, n_str);
        var temp_str_buf: [32]u8 = undefined;
        const temp_str = try std.fmt.bufPrint(&temp_str_buf, "{d}", .{params.temperature});
        try args_buf.append(self.allocator, "--temp");
        try args_buf.append(self.allocator, temp_str);
        if (params.top_k > 0) {
            var k_buf: [32]u8 = undefined;
            const k_str = try std.fmt.bufPrint(&k_buf, "{d}", .{params.top_k});
            try args_buf.append(self.allocator, "--top-k");
            try args_buf.append(self.allocator, k_str);
        }
        if (params.top_p < 1.0) {
            var p_buf: [32]u8 = undefined;
            const p_str = try std.fmt.bufPrint(&p_buf, "{d}", .{params.top_p});
            try args_buf.append(self.allocator, "--top-p");
            try args_buf.append(self.allocator, p_str);
        }
        var seed_buf: [32]u8 = undefined;
        const seed_str = try std.fmt.bufPrint(&seed_buf, "{d}", .{params.seed});
        try args_buf.append(self.allocator, "--seed");
        try args_buf.append(self.allocator, seed_str);
        // -no-cnv: no chat template auto (el prompt ya viene formateado)
        try args_buf.append(self.allocator, "-no-cnv");
        // -ngl 99: forzar offload a GPU si hay
        try args_buf.append(self.allocator, "-ngl");
        try args_buf.append(self.allocator, "99");
        // Silenciar banner de progreso
        try args_buf.append(self.allocator, "--log-disable");

        // Spawn sub-proceso y capturar stdout. argv[0] = engine_bin, resto = args.
        const io = std.Io.Threaded.global_single_threaded.io();
        var argv_buf: std.ArrayList([]const u8) = .empty;
        defer argv_buf.deinit(self.allocator);
        try argv_buf.append(self.allocator, self.engine_bin);
        for (args_buf.items) |a| try argv_buf.append(self.allocator, a);
        var child = try std.process.spawn(io, .{
            .argv = argv_buf.items,
            .stdout = .pipe,
            .stderr = .pipe,
        });
        // wait() debe llamarse antes o después; en 0.16 es el destructor
        // de facto del child. Lo deferimos tras recoger stdout.
        defer {
            _ = child.wait(io) catch {};
            // cerrar file handles de pipe
            if (child.stdout) |*f| f.close(io);
            if (child.stderr) |*f| f.close(io);
        }

        // Leer stdout completo usando readAlloc con realloc dinámico
        const stdout = child.stdout orelse return error.ChildNoStdout;
        const stderr = child.stderr orelse return error.ChildNoStderr;

        var stdout_buf: std.ArrayList(u8) = .empty;
        defer stdout_buf.deinit(self.allocator);
        var read_buf: [4096]u8 = undefined;
        var reader = stdout.reader(io, &read_buf);
        while (true) {
            // readSliceAll falla con EndOfStream al EOF, lo capturamos.
            const chunk = reader.interface.readAlloc(self.allocator, 4096) catch |err| switch (err) {
                error.EndOfStream => break,
                else => return err,
            };
            defer self.allocator.free(chunk);
            try stdout_buf.appendSlice(self.allocator, chunk);
        }

        // Drain stderr
        var err_buf: std.ArrayList(u8) = .empty;
        defer err_buf.deinit(self.allocator);
        var err_read_buf: [4096]u8 = undefined;
        var err_reader = stderr.reader(io, &err_read_buf);
        while (true) {
            const chunk = err_reader.interface.readAlloc(self.allocator, 4096) catch break;
            defer self.allocator.free(chunk);
            try err_buf.appendSlice(self.allocator, chunk);
        }

        const output = try stdout_buf.toOwnedSlice(self.allocator);
        _ = err_buf.items;

        const end = time.Timer.now();
        const elapsed_ns: u64 = if (end > start) @intCast(end - start) else 1;
        const elapsed_ms: u64 = elapsed_ns / std.time.ns_per_ms;

        // Parsear tokens_per_second del output (formato esperado:
        // "Generated N tokens in M ms (T.TT t/s)").
        // Si no, estimamos 1 token = 4 chars como fallback.
        const completion_tokens = countGeneratedTokens(output) catch
            @divFloor(output.len, 4);
        const elapsed_secs: f64 = @as(f64, @floatFromInt(elapsed_ms)) / 1000.0;
        const tps: f32 = @floatCast(@as(f64, @floatFromInt(completion_tokens)) / elapsed_secs);

        const prompt_tokens = self.countPromptTokens(params.prompt);

        return .{
            .text = output,
            .prompt_tokens = prompt_tokens,
            .completion_tokens = completion_tokens,
            .total_ms = elapsed_ms,
            .tokens_per_second = tps,
        };
    }
};

/// Cuenta los tokens generados parseando el output del CLI. El CLI actual
/// no emite una línea "Generated N tokens" consistente, así que este parser
/// es defensivo. Si no encuentra nada, devuelve error y el caller usa
/// heurística de chars/4.
fn countGeneratedTokens(output: []const u8) !usize {
    // Buscar "Generated" seguido de un número
    const idx = std.mem.indexOf(u8, output, "Generated ") orelse return error.NotFound;
    var pos: usize = idx + "Generated ".len;
    var n: usize = 0;
    while (pos < output.len and output[pos] >= '0' and output[pos] <= '9') : (pos += 1) {
        n = n * 10 + (output[pos] - '0');
    }
    return n;
}

/// Spinlock casero sobre un atomic u32. Zig 0.16 no tiene std.Thread.Mutex
/// y queremos evitar POSIX shims para algo tan simple. Sólo se usa 1 vez
/// por request y los spins son O(1) en la práctica.
fn acquireLock(l: *std.atomic.Value(u32)) void {
    while (true) {
        const prev = l.cmpxchgWeak(0, 1, .seq_cst, .seq_cst);
        if (prev == null) return;
        // Spin briefly
        var i: u32 = 0;
        while (i < 100) : (i += 1) std.atomic.spinLoopHint();
    }
}

fn releaseLock(l: *std.atomic.Value(u32)) void {
    _ = l.store(0, .seq_cst);
}

// ─── Tests ───────────────────────────────────────────────────────────────

test "engine construction fails on missing binary" {
    const allocator = std.testing.allocator;
    const result = Engine.init(allocator, "/nonexistent/binary", "/nonexistent/model.gguf");
    try std.testing.expectError(Engine.InitError.EngineBinNotFound, result);
}

test "countGeneratedTokens parses 'Generated 42 tokens'" {
    const out = "...\nGenerated 42 tokens in 1234 ms (34.0 t/s)\n";
    try std.testing.expectEqual(@as(usize, 42), try countGeneratedTokens(out));
}
