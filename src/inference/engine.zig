//! InferenceEngine — la inferencia híbrida dividida en setup persistente
//! (una vez) y generación per-request (N-veces) con TokenSink streaming.
//!
//! T2b del plan server F2. `Engine` reemplaza a `runHybridInference` para
//! el camino del SERVER: carga el modelo una vez, atiende muchas requests
//! vía `generate()`, y emite token-por-token al sink en lugar de acumular.
//!
//! El CLI legacy (cli.zig runHybridInference) NO se toca — sigue siendo el
//! camino byte-idéntico de una sola corrida. Engine comparte TODOS los
//! helpers de cli.zig (movidos a T1) vía import.
//!
//! v1 de este engine: 1 request a la vez (serializado por el BatchingLoop);
//! las secuencias KV se aíslan por scheduler.submit/finishSequence + un
//! paged_kv compartido. Continuous batching multi-secuencia (v2) reusa el
//! scheduler igual que el prefill del CLI lo hace hoy.

const std = @import("std");
const cli = @import("cli");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const pipeline = @import("pipeline");
const gguf_model = @import("gguf_model");
const gguf_tokenizer = @import("gguf_tokenizer");
const bpe = @import("tokenizer");
const cudaz = @import("cudaz");
const cublas = @import("cublas");
const layer_kernels = @import("layer_kernels");
const embedding = @import("embedding");
const hybrid_layer = @import("transformer");
const paged_attn = @import("paged_attention");
const vram_budget = @import("vram_budget");
const debugz = @import("debug");
const specdrv = @import("speculative");
const gguf_moe = @import("gguf_moe");
const moe_layer = @import("moe_layer");
const moe_cuda = @import("moe_cuda");
const offload_cache = @import("offload_cache");
const host_bank = @import("host_bank");
const time = @import("time");

const TokenSinkT = @import("../server/token_sink.zig").TokenSink;
const FinishReason = @import("../server/token_sink.zig").FinishReason;
const Usage = @import("../server/token_sink.zig").Usage;

pub const CliParams = cli.CliParams;
pub const runHybridInference = cli.runHybridInference;

/// Config de generación per-request.
pub const GenerateParams = struct {
    prompt_tokens: []const u32, // BPE ya aplicado por el caller
    max_new_tokens: usize = 256,
    sampler: pipeline.Sampler = .{},
    seed: u64 = 42,
    deadline_ms: u64 = 0, // 0 = sin timeout
    sink: TokenSinkT,
};

/// Error de generación.
pub const GenerateError = error{
    EngineNotReady,
    InvalidPrompt,
    ContextOverflow,
    DeadlineExceeded,
    OutOfMemory,
    CudaError,
};

// ─── Parte 2: bucle de generación reutilizable (extracto del CLI 2610+) ───

/// Genera una secuencia completa usando el Engine montado, emitiendo tokens
/// por el sink. Es la refactorización del bucle de decode del CLI (el CLI
/// legacy sigue igual; este es el camino del server).
///
/// El cuerpo está extraído del decode-loop de runHybridInference (CLI
/// 2610-2882) con estos cambios:
///   1. gen_tokens no se acumula en memoria infinita: emite al sink.
///   2. el sampler viene del request (no de CliParams).
///   3. prompt tokens vienen del request (no del prompt CLI).
///   4. deadline: si wall-clock > deadline_ms, termina con .cancelled.
pub const Engine = struct {
    // Estado montado una vez (los mismos campos que los locals del CLI).
    allocator: std.mem.Allocator,
    model: *gguf_model.GgufModel,
    model_path: []const u8,
    params: CliParams,
    backend: matmul.Backend,

    pub const InitError = error{
        ModelLoadFailed,
        OutOfMemory,
        CudaError,
    };

    /// Monta el engine (carga modelo completo, calienta GPU). El caller
    /// pasa el modelo ya cargado (main.zig / mod.zig lo gestionan).
    pub fn init(
        allocator: std.mem.Allocator,
        model: *gguf_model.GgufModel,
        model_path: []const u8,
        params: CliParams,
        backend: matmul.Backend,
    ) InitError!Engine {
        _ = allocator;
        _ = model;
        _ = model_path;
        _ = params;
        _ = backend;
        return error.EngineNotReady; // T2c: montaje completo del estado
    }

    /// Genera una secuencia. No-streaming-ready: cada token emitido al sink.
    pub fn generate(self: *Engine, gen_params: GenerateParams) !Usage {
        _ = self;
        _ = gen_params;
        return error.EngineNotReady; // T2c: loop de decode con sink
    }

    /// Shutdown: libera todo el estado montado.
    pub fn deinit(self: *Engine) void {
        _ = self;
    }
};
