//! GraphCapture — CUDA Graph capture refactor (Phase 3, FreeToken Technique 9).
//!
//! Máquina de estados reutilizable para capturar una secuencia de
//! kernels/copias de un stream en un CUDA graph re-ejecutable
//! (`cuGraphLaunch`), eliminando el overhead de dispatch por kernel
//! (decode: ~290 lanzamientos → 1).
//!
//! Módulo mínimo (solo cudaz + debug) para que puedan componerlo tanto
//! matmul (`matmul.GraphCapture`) como decode_graph (DecodeGraph) sin
//! dependencias cruzadas.
//!
//! Principios graph-safe de FreeToken (Section 11) que hace cumplir:
//!   1. Descriptores de copia fijos: los nodos HtoD capturados copian SIEMPRE
//!      del mismo staging host pineado al mismo buffer device; solo cambia el
//!      CONTENIDO del staging entre replays (nunca los punteros).
//!   2. Reset tras captura: el run de captura muta estado persistente (KV,
//!      recurrencia ssm, contadores on-device); el caller respalda/restaura
//!      por su cuenta (ver DecodeGraph.backupState/restoreState).
//!   3. Abort garantizado: si un nodo falla dentro de la captura, `abort()`
//!      termina la captura (destruyendo la definición válida si la hubiera,
//!      SIN leak) y devuelve el stream a modo normal (auto-fallback al
//!      camino lanzado kernel a kernel).
//!   4. Cero syncs dentro: `cuStreamSynchronize`/`cudaDeviceSynchronize`
//!      durante la captura la invalidan.
const std = @import("std");
const cudaz = @import("cudaz");
const debugz = @import("debug");

pub const GraphCapture = struct {
    stream: cudaz.CUstream,
    /// Instancia re-ejecutable; null hasta el primer capture exitoso.
    exec: ?cudaz.CUgraphExec = null,
    /// Nodos del último grafo instanciado (breadcrumb DUMP_GRAPH).
    last_node_count: usize = 0,
    /// Generación: +1 por cada re-captura exitosa.
    generation: u64 = 0,

    const Self = @This();

    pub fn init(stream: cudaz.CUstream) Self {
        return .{ .stream = stream };
    }

    pub fn deinit(self: *Self) void {
        if (self.exec) |e| {
            cudaz.cuGraphExecDestroy(e);
            self.exec = null;
        }
    }

    pub fn isReplayable(self: *const Self) bool {
        return self.exec != null;
    }

    /// Empieza la captura GLOBAL del stream (captura todo el trabajo CUDA
    /// en él). El caller lanza la secuencia con sus APIs normales y llama
    /// `finish()` (éxito) o `abort()` (error).
    pub fn begin(self: *Self) !void {
        try cudaz.cuStreamBeginCapture(self.stream, .GLOBAL);
    }

    /// Termina la captura y crea la instancia re-ejecutable. Devuelve el
    /// conteo de nodos capturados. En error, el stream ya vuelve a modo
    /// normal y `exec` queda intacto (replay del grafo anterior si lo había).
    pub fn finish(self: *Self) !usize {
        const graph = try cudaz.cuStreamEndCapture(self.stream);
        errdefer cudaz.cuGraphDestroy(graph);
        var node_count: usize = 0;
        if (cudaz.cuGraphGetNodeCount(graph)) |n| {
            node_count = n;
        } else |_| {}
        var exec: cudaz.CUgraphExec = undefined;
        try cudaz.cuGraphInstantiateWithParams(&exec, graph, cudaz.CUDA_GRAPH_INSTANTIATE_FLAG_DEFAULT);
        // Éxito: sustituir la instancia anterior (si había).
        if (self.exec) |old| cudaz.cuGraphExecDestroy(old);
        self.exec = exec;
        self.last_node_count = node_count;
        self.generation += 1;
        cudaz.cuGraphDestroy(graph);
        if (debugz.dbg.dump_graph) {
            debugz.dbg.printLevel(.info, "[graph] capturado gen={d} nodes={d}\n", .{ self.generation, node_count });
        }
        return node_count;
    }

    /// Aborta la captura en curso (nodo fallido). Siempre deja el stream en
    /// modo normal. CUDA requiere consumir el EndCapture para volver al modo
    /// lanzable; si el EndCapture devuelve un grafo válido igual lo
    /// destruye (SIN leak de la definición). Idempotente: si la captura ya
    /// terminó, es no-op.
    pub fn abort(self: *Self) void {
        if (cudaz.cuStreamEndCapture(self.stream)) |graph| {
            // Captura válida que no queremos: destruir la definición.
            cudaz.cuGraphDestroy(graph);
        } else |_| {}
        // Si la captura quedó invalidada (nodo erróneo), un segundo
        // EndCapture la limpia; si ya estaba limpia, devuelve error no-op.
        _ = cudaz.cuStreamEndCapture(self.stream) catch {};
    }

    /// Re-ejecuta el grafo capturado (un launch = toda la secuencia).
    pub fn replay(self: *Self) !void {
        try cudaz.cuGraphLaunch(self.exec.?, self.stream);
    }
};

// ── Tests (máquina de estados pura, sin GPU) ─────────────────────────────────

const testing = std.testing;

test "GraphCapture: estado inicial no replayable, deinit sin exec es no-op" {
    var gc = GraphCapture.init(@as(cudaz.CUstream, @ptrFromInt(0xdead)));
    defer gc.deinit();
    try testing.expectEqual(false, gc.isReplayable());
    try testing.expectEqual(@as(usize, 0), gc.last_node_count);
    try testing.expectEqual(@as(u64, 0), gc.generation);
}
