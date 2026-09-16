//! BatchingLoop — hilo worker del engine con continuous batching.
//!
//! T2 del plan server F2. El loop vive en su propio hilo (std.Thread.spawn,
//! disponible en 0.16) y es el ÚNICO que toca el InferenceEngine
//! (thread-safety por aislamiento: los handlers HTTP sólo hacen
//! enqueue/submit y leen el sink).
//!
//! Modelo de scheduling v1 (documentado en SERVER_F2_PLAN.md):
//!   - 1 secuencia GPU activa (decode serial; CUDA graphs siguen
//!     funcionando igual que el CLI).
//!   - Requests en espera: cola FIFO con backpressure (503 al superar
//!     max_queue_depth).
//!   - El prefill de la siguiente request se solapa con el decode en curso
//!     cuando el engine lo permita (v2: prefill chunked entre tokens).
//!
//! La cola usa un spinlock (igual patrón que engine.zig Fase 1, probado);
//! la contención es mínima: 1 append + 1 popleft por request completo.

const std = @import("std");
const debugz = @import("debug");
const token_sink = @import("token_sink");
const time = @import("time");

pub const TokenSink = token_sink.TokenSink;
pub const FinishReason = token_sink.FinishReason;
pub const Usage = token_sink.Usage;

/// Request en cola: prompt ya tokenizado + sampling + sink del cliente.
pub const PendingRequest = struct {
    req_id: u64,
    /// Tokens del prompt (BPE ya aplicado; ownership: el loop lo libera).
    prompt_tokens: []u32,
    max_new_tokens: usize,
    seed: u64,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    /// Deadline en ms (wall clock); 0 = sin timeout.
    deadline_ms: u64 = 0,
    /// Sink que recibe los tokens (vtable del protocolo: SSE/NDJSON/collect).
    sink: TokenSink,
};

/// Errores de submit.
pub const SubmitError = error{
    QueueFull,
    Shutdown,
    OutOfMemory,
};

/// Estado de vida del loop.
const State = enum { starting, ready, running, draining, stopped };

pub const BatchingLoop = struct {
    allocator: std.mem.Allocator,
    /// Cola FIFO protegida por spinlock.
    queue: std.ArrayList(PendingRequest),
    lock: std.atomic.Value(u32) = .{ .raw = 0 },
    state: std.atomic.Value(u8) = .{ .raw = @intFromEnum(State.starting) },
    /// Backpressure: máx requests encoladas esperando (503 al superar).
    max_queue_depth: usize = 64,
    next_req_id: u64 = 1,
    /// Contadores para /health y métricas.
    submitted: std.atomic.Value(u64) = .{ .raw = 0 },
    completed: std.atomic.Value(u64) = .{ .raw = 0 },
    rejected: std.atomic.Value(u64) = .{ .raw = 0 },

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, max_queue_depth: usize) Self {
        return .{
            .allocator = allocator,
            .queue = .empty,
            .max_queue_depth = max_queue_depth,
        };
    }

    pub fn deinit(self: *Self) void {
        // Libera los tokens de los requests no consumidos (ownership).
        for (self.queue.items) |req| self.allocator.free(req.prompt_tokens);
        self.queue.deinit(self.allocator);
    }

    /// Encola un request. Los prompt_tokens pasan a ser ownership del loop
    /// (se liberan al consumir). Devuelve el req_id asignado.
    pub fn submit(self: *Self, prompt_tokens: []u32, sink: TokenSink, opts: struct {
        max_new_tokens: usize = 256,
        seed: u64 = 42,
        temperature: f32 = 0.7,
        top_k: usize = 40,
        top_p: f32 = 0.95,
        deadline_ms: u64 = 120_000,
    }) SubmitError!u64 {
        if (self.stateValue() == .stopped or self.stateValue() == .draining)
            return SubmitError.Shutdown;
        acquire(&self.lock);
        defer release(&self.lock);
        if (self.queue.items.len >= self.max_queue_depth) {
            _ = self.rejected.fetchAdd(1, .monotonic);
            return SubmitError.QueueFull;
        }
        const req = PendingRequest{
            .req_id = self.next_req_id,
            .prompt_tokens = prompt_tokens,
            .max_new_tokens = opts.max_new_tokens,
            .seed = opts.seed,
            .temperature = opts.temperature,
            .top_k = opts.top_k,
            .top_p = opts.top_p,
            .deadline_ms = opts.deadline_ms,
            .sink = sink,
        };
        self.queue.append(self.allocator, req) catch {
            self.allocator.free(prompt_tokens);
            return SubmitError.OutOfMemory;
        };
        self.next_req_id += 1;
        _ = self.submitted.fetchAdd(1, .monotonic);
        return req.req_id;
    }

    fn stateValue(self: *Self) State {
        return @enumFromInt(self.state.load(.acquire));
    }

    /// Marca el loop para drain: no acepta nuevos, termina los encolados.
    pub fn beginDrain(self: *Self) void {
        _ = self.state.cmpxchgStrong(
            @intFromEnum(State.running),
            @intFromEnum(State.draining),
            .acq_rel,
            .acquire,
        );
    }

    /// Cantidad en cola (aprox; para /health).
    pub fn queueDepth(self: *Self) usize {
        acquire(&self.lock);
        defer release(&self.lock);
        return self.queue.items.len;
    }

    // El pop es interno (sólo el hilo del loop tras el arranque).
    fn pop(self: *Self) ?PendingRequest {
        acquire(&self.lock);
        defer release(&self.lock);
        if (self.queue.items.len == 0) return null;
        const req = self.queue.orderedRemove(0);
        return req;
    }

    /// Cuerpo del hilo. `engine_run` ejecuta UNA secuencia completa:
    /// fn(prompt_tokens, sampling, sink) → Usage/FnErr. Lo inyecta el
    /// caller porque el InferenceEngine concreto (CLI-legacy re-entrado
    /// vs. engine puro) se conecta en mod.zig según T2.
    pub fn run(
        self: *Self,
        engine_run: *const fn (PendingRequest) EngineError!Usage,
    ) void {
        _ = self.state.cmpxchgStrong(
            @intFromEnum(State.starting),
            @intFromEnum(State.running),
            .acq_rel,
            .acquire,
        );
        debugz.dbg.printLevel(.info, "[server] BatchingLoop: iniciado\n", .{});
        while (true) {
            const st = self.stateValue();
            if (st == .draining and self.queueDepth() == 0) break;
            if (st == .stopped) break;
            const req = self.pop() orelse {
                // Cola vacía: yield corto (no spin del core).
                std.Thread.yield() catch {};
                continue;
            };
            const t0 = time.Timer.now();
            const usage_or_err = engine_run(req);
            if (usage_or_err) |u| {
                req.sink.finish(.stop, u);
                _ = self.completed.fetchAdd(1, .monotonic);
            } else |e| {
                debugz.dbg.printLevel(.info, "[server] seq {d} error: {s}\n", .{ req.req_id, @errorName(e) });
                const elapsed_ms: u64 = @intCast(@divTrunc(time.Timer.now() - t0, std.time.ns_per_ms));
                req.sink.finish(.err, .{
                    .prompt_tokens = req.prompt_tokens.len,
                    .completion_tokens = 0,
                    .total_ms = elapsed_ms,
                });
            }
            self.allocator.free(req.prompt_tokens);
        }
        _ = self.state.swap(@intFromEnum(State.stopped), .release);
        debugz.dbg.printLevel(.info, "[server] BatchingLoop: detenido (completados={d})\n", .{self.completed.load(.monotonic)});
    }

    pub const EngineError = anyerror;
};

// ─── Spinlock (0.16 sin std.Thread.Mutex; mismo patrón probado engine.zig) ──

fn acquire(l: *std.atomic.Value(u32)) void {
    while (true) {
        if (l.cmpxchgWeak(0, 1, .seq_cst, .seq_cst) == null) return;
        var i: u32 = 0;
        while (i < 100) : (i += 1) std.atomic.spinLoopHint();
    }
}

fn release(l: *std.atomic.Value(u32)) void {
    _ = l.store(0, .seq_cst);
}

// ─── Tests ───────────────────────────────────────────────────────────────

test "submit enqueues with backpressure" {
    const allocator = std.testing.allocator;
    var loop = BatchingLoop.init(allocator, 2);
    defer loop.deinit();

    var sink_ctx = token_sink.CollectingSink.init(allocator);
    defer sink_ctx.deinit();

    // Ownership de los tokens pasa al loop (los libera en pop/run); el
    // test sólo libera los de la submission rechazada.
    const toks1 = try allocator.dupe(u32, &[_]u32{ 1, 2, 3 });
    const id1 = try loop.submit(toks1, sink_ctx.sink(), .{});
    try std.testing.expectEqual(@as(u64, 1), id1);

    const toks2 = try allocator.dupe(u32, &[_]u32{4});
    const id2 = try loop.submit(toks2, sink_ctx.sink(), .{});
    try std.testing.expectEqual(@as(u64, 2), id2);

    // Queue llena (max 2): la 3ª rechazada. El submit NO consume los
    // tokens en el rechazo (el caller los retiene y libera).
    const toks3 = try allocator.dupe(u32, &[_]u32{5});
    defer allocator.free(toks3);
    try std.testing.expectError(SubmitError.QueueFull, loop.submit(toks3, sink_ctx.sink(), .{}));
    try std.testing.expectEqual(@as(usize, 2), loop.queueDepth());
    try std.testing.expectEqual(@as(u64, 1), loop.rejected.load(.monotonic));
}
