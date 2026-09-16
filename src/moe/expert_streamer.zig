//! expert_streamer — 4.10: streaming expert-por-expert con doble slot (Gap 3).
//!
//! Productor-consumidor sobre la primitiva FetchStream (4.8 A2): mientras la
//! GPU computa el experto j, el fetch-stream transfiere el j+1 ⇒ el PCIe de
//! los pesos deja de serializarse contra el compute. Estilo ktransformers
//! `fp8_layerwise_transport.cpp:209-710` (double-buffered slots, control
//! por eventos) adaptado a nuestros bancos de slot cache:
//!
//!   - slot COMPUTE (idx `compute_idx`): los GEMM del experto j lo leen.
//!     Su fetch completó (host-sync vía `cuEventSynchronize` en `advance`)
//!     ANTES de lanzar los GEMM ⇒ sin bridge necesario para el promocionado.
//!   - slot TRANSFER (idx `compute_idx^1`): el fetch del experto j+1 corre
//!     en el FetchStream — solapa con los GEMM del j (streams distintos).
//!
//! Protección de carrera (el matiz que hace correcto el ping-pong): el
//!   fetch del experto j+2 se stagea en el slot que el GEMM del j ACABA de
//!   leer ⇒ el fetch-stream espera (`cuStreamWaitEvent`) el event
//!   `compute_done[slot]` que el caller graba tras los GEMM de cada experto
//!   (`markComputeDone`). Sin esto, el memcpy puede pisar bytes en vuelo.
//!
//! Opt-in por proceso (MOE_EXPERT_STREAM=1): el modo reemplaza el
//! ensure+gather de lote del pool LRU por este consumidor por-experto con
//! 2 slots propios (no toca el estado del OffloadCache). Breadcrumbs:
//! MOE_DEBUG=1 + DEBUG_LEVEL (debug.dbg), tag [moe_stream].

const std = @import("std");
const cudaz = @import("cudaz");
const debugz = @import("debug");
const moe_cuda = @import("moe_cuda");

fn streamEnabled() bool {
    return std.c.getenv("MOE_EXPERT_STREAM") != null;
}

pub const Error = error{
    NoExpertInFlight,
};

/// Estado del ping-pong.
pub const StreamState = enum(u8) {
    idle = 0,
    /// Fetch del experto j+1 en vuelo en el FetchStream.
    transferring = 1,
};

/// Streaming 2-slot de la lista top-k de expertos de UNA capa MoE. Los GEMM
/// del experto "actual" corren contra `slotCompute()`; el streamer mantiene
/// el siguiente transfiriéndose en el otro slot.
pub const ExpertStreamer = struct {
    fs: *moe_cuda.FetchStream,
    /// Slots ping-pong (índices de fila del cache destino, por banco).
    slot_ids: [2]i32,
    /// Índice (0/1) del slot de compute ACTUAL (toggle por advance).
    compute_idx: usize = 0,
    state: StreamState = .idle,
    /// Experto residente en el slot de compute (tras el primer advance).
    current_expert: i32 = -1,
    /// Experto cuyo fetch está en vuelo (slot de transfer).
    pending_expert: i32 = -1,
    /// Events "compute terminó de leer el slot i": ev[i] protege
    /// slot_ids[i]. El fetch-stream espera el del slot que va a pisar.
    compute_done: [2]cudaz.CUevent,
    ev_recorded: [2]bool = .{ false, false },

    /// Geometría de bancos de la capa (idéntica a gatherMissing):
    /// dst_bases/src_bases/feat_bytes paralelos por banco.
    dst_bases: []const usize,
    src_bases: []const usize,
    feat_bytes: []const usize,

    const Self = @This();

    /// `slot_a`/`slot_b`: filas del cache destino para el ping-pong (el
    /// caller las elige FUERA del alcance del LRU del pool — en modo
    /// streaming el pool no corre).
    pub fn init(
        fs: *moe_cuda.FetchStream,
        slot_a: i32,
        slot_b: i32,
        dst_bases: []const usize,
        src_bases: []const usize,
        feat_bytes: []const usize,
    ) !Self {
        const ev0 = try cudaz.cuEventCreate(0);
        errdefer cudaz.cuEventDestroy(ev0);
        const ev1 = try cudaz.cuEventCreate(0);
        errdefer cudaz.cuEventDestroy(ev1);
        return .{
            .fs = fs,
            .slot_ids = .{ slot_a, slot_b },
            .compute_done = .{ ev0, ev1 },
            .dst_bases = dst_bases,
            .src_bases = src_bases,
            .feat_bytes = feat_bytes,
        };
    }

    pub fn deinit(self: *Self) void {
        cudaz.cuEventDestroy(self.compute_done[0]);
        cudaz.cuEventDestroy(self.compute_done[1]);
    }

    /// Slot donde el compute DEBE leer el experto actual.
    pub fn slotCompute(self: *const Self) i32 {
        return self.slot_ids[self.compute_idx];
    }

    /// Arranca el fetch del primer experto al slot de transfer. El caller
    /// debe tener el stream de compute sincronizado (ningún GEMM previo
    /// puede estar leyendo ese slot aún).
    pub fn begin(self: *Self, first_expert: i32) !void {
        if (first_expert < 0) return;
        try self.stageExpert(first_expert, self.slot_ids[1]);
        self.pending_expert = first_expert;
        self.state = .transferring;
        try self.fs.markReady();
    }

    /// Espera el fetch en vuelo (host), promociona su slot a compute
    /// (swap) y — si `next_expert >= 0` — stagea el fetch siguiente en el
    /// slot liberado (esperando antes el event de compute que lo protege).
    /// Devuelve el slot donde leer el experto promovido.
    ///
    /// Cadencia del caller (solape real):
    ///   begin(e0); loop j { slot = advance(e_{j+1}); GEMM(e_j, slot);
    ///                       markComputeDone(compute); } wait();
    pub fn advance(self: *Self, next_expert: i32) !i32 {
        if (self.state == .transferring) {
            try self.fs.sync();
            self.state = .idle;
        }
        if (self.pending_expert < 0) return Error.NoExpertInFlight;

        // Swap: el slot de transfer (bytes del experto promovido YA en
        // device por el host-sync de arriba) pasa a compute.
        self.compute_idx ^= 1;
        self.current_expert = self.pending_expert;
        self.pending_expert = -1;

        if (next_expert >= 0) {
            // El slot que queda de transfer es el ANTERIOR de compute: el
            // GEMM del experto promovido en la advance PREVIA lo leyó ⇒ el
            // fetch espera su event antes de pisarlo.
            const tidx = self.compute_idx ^ 1;
            if (self.ev_recorded[tidx])
                try cudaz.cuStreamWaitEvent(self.fs.stream, self.compute_done[tidx], 0);
            try self.stageExpert(next_expert, self.slot_ids[tidx]);
            self.pending_expert = next_expert;
            self.state = .transferring;
            try self.fs.markReady();
        }
        if (moeDebugOn() and debugz.dbg.at(.detail))
            debugz.dbg.print("[moe_stream] advance: current=e{d} slot={d} pending=e{d} state={s}\n", .{ self.current_expert, self.slot_ids[self.compute_idx], self.pending_expert, @tagName(self.state) });
        return self.slot_ids[self.compute_idx];
    }

    /// Graba el event "compute terminó con el slot actual" — llamar tras
    /// lanzar los GEMM del experto promovido (cierra la protección del
    /// ping-pong para el fetch que lo pisará dentro de 2 advances).
    pub fn markComputeDone(self: *Self, compute: cudaz.CUstream) !void {
        try cudaz.cuEventRecord(self.compute_done[self.compute_idx], compute);
        self.ev_recorded[self.compute_idx] = true;
    }

    /// Espera dura el fetch en vuelo (fin de la ronda — antes de leer
    /// dev_acc o reutilizar los slots).
    pub fn wait(self: *Self) !void {
        if (self.state == .transferring) {
            try self.fs.sync();
            self.state = .idle;
        }
    }

    fn stageExpert(self: *const Self, expert: i32, slot: i32) !void {
        const e: usize = @intCast(expert);
        const s: usize = @intCast(slot);
        for (0..self.feat_bytes.len) |b| {
            self.fs.stageHtoD(
                self.dst_bases[b] + s * self.feat_bytes[b],
                self.src_bases[b] + e * self.feat_bytes[b],
                self.feat_bytes[b],
            ) catch |err| {
                // PROPA GAR: un stage fallido dejaría bytes stale del experto
                // previo en el slot ⇒ salida incorrecta SILenciosa. El error
                // sube al caller (advance/begin) y de ahí al fallback de
                // moe_layer (camino batch).
                if (moeDebugOn())
                    debugz.dbg.print("[moe_stream] stageHtoD failed banco {d}: {s}\n", .{ b, @errorName(err) });
                return err;
            };
        }
    }
};

/// Gate global del streaming por-experto (MOE_EXPERT_STREAM=1).
pub fn enabled() bool {
    if (g_force_for_test != null) return g_force_for_test.?;
    return streamEnabled();
}

/// Tests (test_moe_layer_gpu): fuerza el gate sin depender del orden de
/// tests del binario ni de mutar el environment del proceso.
pub fn forceForTest(on: bool) void {
    g_force_for_test = on;
}

var g_force_for_test: ?bool = null;

fn moeDebugOn() bool {
    return std.c.getenv("MOE_DEBUG") != null;
}

// ── Tests (GPU) ──────────────────────────────────────────────────────────────
// La validación real (paridad de bytes + cadencia del ping-pong) vive en
// tests/test_moe_expert_stream_gpu.zig: requiere FetchStream/CUDA. El FSM
// (swap/events/waits) es demasiado fino para probarse sin los streams.
