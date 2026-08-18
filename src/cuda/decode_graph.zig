//! CUDA Graphs para el decode por token de la capa híbrida.
//!
//! La secuencia de un token (embed H2D + 24 capas híbridas + rmsNorm final +
//! lm_head, ~290 lanzamientos) se captura UNA vez y se reproduce con un solo
//! `cuGraphLaunch`, eliminando el overhead de dispatch por kernel.
//!
//! Los únicos valores que cambian por token (embedding, block table, start_pos,
//! seq_len) viven en staging buffers host persistentes que los nodos capturados
//! (HtoDAsync) copian a device; los kernels leen esos valores de buffers device
//! fijos. Así el grafo es totalmente estático: sin updates por nodo ni
//! re-captura.
//!
//! Estado persistente que el run de captura corrompe (estado de recurrencia
//! DeltaNet y conv_state de los ssm): se respalda en un buffer device antes de
//! capturar y se restaura después (D2D). El KV cache no necesita respaldo: el
//! token 1 reescribe exactamente la posición que escribió la captura.
const std = @import("std");
const cudaz = @import("cudaz");

pub const Mode = enum { off, replay };

/// Una porción de estado persistente de capa ssm a respaldar/restaurar.
pub const StatePart = struct {
    dev: cudaz.CUdeviceptr,
    bytes: usize,
};

pub const DecodeGraph = struct {
    allocator: std.mem.Allocator,
    stream: cudaz.CUstream,
    n_embd: usize,
    /// Staging host del embedding (f32, persistente, lo aloca el llamador); los
    /// nodos capturados copian a g_cur.
    embed_staging: []f32 = &.{},
    exec: ?cudaz.CUgraphExec = null,
    /// Respaldo del estado ssm (device): buf + total.
    state_bak: cudaz.CUdeviceptr = 0,
    state_bak_bytes: usize = 0,

    pub fn init(allocator: std.mem.Allocator, stream: cudaz.CUstream, n_embd: usize) !DecodeGraph {
        return .{
            .allocator = allocator,
            .stream = stream,
            .n_embd = n_embd,
        };
    }

    pub fn deinit(self: *DecodeGraph) void {
        if (self.exec) |e| cudaz.cuGraphExecDestroy(e);
        if (self.state_bak != 0) cudaz.cuMemFree(self.state_bak);
    }

    /// Fija el staging host del embedding (persistente, lo aloca el llamador y
    /// lo usa también como fuente del H2D en el camino normal).
    pub fn setEmbedStaging(self: *DecodeGraph, buf: []f32) void {
        self.embed_staging = buf;
    }

    pub fn mode(self: *const DecodeGraph) Mode {
        return if (self.exec != null) .replay else .off;
    }

    /// Lanza el grafo (toda la secuencia GPU de un token) en el stream.
    pub fn launch(self: *DecodeGraph) !void {
        try cudaz.cuGraphLaunch(self.exec.?, self.stream);
    }

    /// Respalda el estado ssm persistente en un buffer device propio.
    pub fn backupState(self: *DecodeGraph, parts: []const StatePart) !void {
        if (self.state_bak != 0) cudaz.cuMemFree(self.state_bak);
        var total: usize = 0;
        for (parts) |p| total += p.bytes;
        if (total == 0) return;
        const buf = try cudaz.cuMemAlloc(total);
        errdefer cudaz.cuMemFree(buf);
        var off: usize = 0;
        for (parts) |p| {
            try cudaz.cuMemcpyDtoD(buf + off, p.dev, p.bytes);
            off += p.bytes;
        }
        self.state_bak = buf;
        self.state_bak_bytes = total;
    }

    /// Restaura el estado ssm desde el respaldo (D2D) y lo libera.
    pub fn restoreState(self: *DecodeGraph, parts: []const StatePart) !void {
        if (self.state_bak == 0) return;
        var off: usize = 0;
        for (parts) |p| {
            try cudaz.cuMemcpyDtoD(p.dev, self.state_bak + off, p.bytes);
            off += p.bytes;
        }
        cudaz.cuMemFree(self.state_bak);
        self.state_bak = 0;
        self.state_bak_bytes = 0;
    }

    /// Empieza la captura del stream (modo GLOBAL: captura todo el trabajo CUDA).
    pub fn beginCapture(self: *DecodeGraph) !void {
        try cudaz.cuStreamBeginCapture(self.stream, .GLOBAL);
    }

    /// Termina la captura y devuelve el grafo definido; null si la captura
    /// falló (un nodo lanzado dentro falló). En cualquier caso el stream vuelve
    /// a modo normal: un segundo `cuStreamEndCapture` limpia la captura
    /// abortada, necesario para el auto-fallback al camino normal.
    pub fn endCapture(self: *DecodeGraph) ?cudaz.CUgraph {
        if (cudaz.cuStreamEndCapture(self.stream)) |graph| {
            return graph;
        } else |_| {
            _ = cudaz.cuStreamEndCapture(self.stream) catch null;
            return null;
        }
    }

    /// Termina la captura, instancia el grafo y descarta la definición.
    /// En error, `exec` queda null (el llamador cae al camino normal).
    pub fn endCaptureAndInstantiate(self: *DecodeGraph) !void {
        const graph = self.endCapture() orelse return error.CaptureFailed;
        errdefer cudaz.cuGraphDestroy(graph);
        var exec: cudaz.CUgraphExec = undefined;
        try cudaz.cuGraphInstantiateWithParams(&exec, graph, cudaz.CUDA_GRAPH_INSTANTIATE_FLAG_DEFAULT);
        cudaz.cuGraphDestroy(graph);
        self.exec = exec;
    }
};