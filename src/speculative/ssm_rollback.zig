//! Ring multi-slot de snapshots del estado recurrente (G6-rem, lane-b1 2026-09-08).
//!
//! Objetivo: en speculative decoding sobre modelos SSM (Qwen3.5/3.8 DeltaNet,
//! LFM2.5 ShortConv), el rechazo de un draft exige restaurar el estado
//! recurrente (S-state + conv-state) a la posición commitida. El rollback
//! básico (1-slot, lane-c C4.3 @6deb99d) solo restaura al ÚLTIMO punto
//! commitido: suficiente para una ronda. Este ring extiende a K slots para
//! permitir rollback a CUALQUIER posición de la ronda (equivalente host-side
//! de `keep_rs_t`/`n_rs_seq` de llama.cpp gated_delta_net.cu:145-157) — útil
//! cuando el verify acepta parcialmente (m < kq) y el siguiente ancla debe
//! partir de un estado intermedio.
//!
//! Modelo de slots (espejo de llama.cpp): slot 0 = estado MÁS RECIENTE,
//! slot s = s tokens atrás. Al snapshotear en la posición `abs` de la ronda,
//! el slot se calcula como `(round_len - 1 - local_idx)` donde `local_idx` es
//! la posición dentro del bloque de tokens de la ronda actual. Si la ronda
//! tiene menos tokens que el ring (K), los slots viejos quedan sin sobrescribir
//! (caller-owned, igual que llama.cpp: "older slots are caller-owned").
//!
//! La primitiva (snapshot/restore) la provee el caller vía `Hooks` — reutiliza
//! `SsmLayer.snapshotGpuState`/`restoreGpuState` y `ShortConvLayer.*`, sin
//! duplicar el layout (d_s_state + d_conv_state en un solo []f32 plano).
//!
//! Uso:
//! ```zig
//! var ring = try SsmRollback.init(allocator, .{ .k = 4, .state_len = n });
//! defer ring.deinit(allocator);
//! // por cada posición local de la ronda:
//! try ring.push(hooks_snapshot, hooks, local_idx, round_len); // snapshot al slot correspondiente
//! // al rechazar (el ancla siguiente debe partir de la posición aceptada m):
//! try ring.restoreAt(hooks_restore, hooks, accepted_local);   // restore al slot de la posición aceptada
//! ```

const std = @import("std");

pub const Config = struct {
    /// Número de slots del ring (K). Slot 0 = más reciente.
    k: usize = 4,
    /// Elementos f32 por snapshot (gpuStateLen de la capa).
    state_len: usize = 0,
};

pub const Hooks = struct {
    /// snapshotGpuState-equivalente: copia el estado device → `dst` (host).
    snapshot: *const fn (ctx: *anyopaque, dst: []f32) anyerror!void,
    /// restoreGpuState-equivalente: restaura `src` (host) → device.
    restore: *const fn (ctx: *anyopaque, src: []const f32) anyerror!void,
    ctx: *anyopaque,
};

pub const SsmRollback = struct {
    allocator: std.mem.Allocator,
    slots: []f32, // k * state_len, contiguo
    config: Config,
    /// Cuántos slots válidos hay en la ronda actual (≤ k). Al inicio de cada
    /// ronda se resetea a 0.
    valid: usize = 0,
    /// Posición local (dentro de la ronda) del snapshot más reciente.
    max_local: usize = 0,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: Config) !Self {
        const total = try std.math.mul(usize, config.k, config.state_len);
        const slots = try allocator.alloc(f32, total);
        return .{
            .allocator = allocator,
            .slots = slots,
            .config = config,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.slots);
        self.* = undefined;
    }

    /// Resetea el ring al inicio de cada ronda de spec (los slots se reutilizan).
    pub fn reset(self: *Self) void {
        self.valid = 0;
        self.max_local = 0;
    }

    /// Snapshot del estado en la posición local `local_idx` de un bloque de
    /// `round_len` tokens. Slot 0 = estado MÁS RECIENTE (local_idx ==
    /// round_len-1); slot dist = dist tokens atrás. Espejo de llama.cpp
    /// `target_slot = n_tokens-1-t`. Si dist >= k el estado es demasiado
    /// viejo para el ring → RingOverflow (el caller decide).
    pub fn push(
        self: *Self,
        h: Hooks,
        local_idx: usize,
        round_len: usize,
    ) !usize {
        const dist = round_len - 1 - local_idx; // 0 = más reciente
        if (dist >= self.config.k) return error.RingOverflow;
        const slot = dist; // slot 0 = reciente
        const base = slot * self.config.state_len;
        try h.snapshot(h.ctx, self.slots[base .. base + self.config.state_len]);
        self.valid = @max(self.valid, slot + 1);
        self.max_local = @max(self.max_local, local_idx);
        return slot;
    }

    /// Restaura el estado al slot de la posición local `local_idx` (el ancla
    /// siguiente tras aceptar `local_idx+1` tokens). Si el slot no fue
    /// escrito esta ronda, devuelve false (el caller decide).
    pub fn restoreAt(self: *Self, h: Hooks, local_idx: usize, round_len: usize) !bool {
        const dist = round_len - 1 - local_idx;
        if (dist >= self.config.k) return false;
        const slot = dist;
        if (slot >= self.valid) return false; // slot no escrito esta ronda
        const base = slot * self.config.state_len;
        try h.restore(h.ctx, self.slots[base .. base + self.config.state_len]);
        return true;
    }

    /// Índice del slot más reciente (siempre 0 cuando hay algún estado).
    pub fn latestSlot(self: *const Self) ?usize {
        if (self.valid == 0) return null;
        return 0;
    }
};

// ─── Tests ────────────────────────────────────────────────────────────────
const test_allocator = std.testing.allocator;

const FakeState = struct {
    data: [24]f32,
};
fn snapFake(ctx: *anyopaque, dst: []f32) anyerror!void {
    const fs: *FakeState = @ptrCast(@alignCast(ctx));
    @memcpy(dst, fs.data[0..dst.len]);
}
fn restoreFake(ctx: *anyopaque, src: []const f32) anyerror!void {
    const fs: *FakeState = @ptrCast(@alignCast(ctx));
    @memcpy(fs.data[0..src.len], src);
}

test "SsmRollback: push restaura el estado exacto de esa posición (espejo llama.cpp)" {
    var fs = FakeState{ .data = undefined };
    var ring = try SsmRollback.init(test_allocator, .{ .k = 4, .state_len = 16 });
    defer ring.deinit();

    const hooks = Hooks{ .snapshot = snapFake, .restore = restoreFake, .ctx = &fs };

    // Ronda de 3 tokens (round_len=3). Snapshot en las posiciones locales 0,1,2.
    // Slot mapeo (0=reciente, slot=dist): pos 2→slot 0, pos 1→slot 1, pos 0→slot 2.
    // Los slots reflejan el estado EXACTO que cada posición veía.
    for (0..3) |li| {
        for (0..16) |i| fs.data[i] = @floatFromInt(li * 100 + i);
        _ = try ring.push(hooks, li, 3);
    }
    try std.testing.expectEqual(@as(usize, 3), ring.valid);

    // Restaurar a la posición local 1 (aceptamos 2 de 3): el estado debe
    // ser el que veía la posición 1 (valores 100..), NO el más reciente (2xx).
    fs.data[0] = -999.0;
    const ok = try ring.restoreAt(hooks, 1, 3);
    try std.testing.expect(ok);
    try std.testing.expectEqual(@as(f32, 100.0), fs.data[0]);
    try std.testing.expectEqual(@as(f32, 115.0), fs.data[15]);
}

test "SsmRollback: restoreAt a posición reciente (0 desde más reciente) da el último" {
    var fs = FakeState{ .data = undefined };
    var ring = try SsmRollback.init(test_allocator, .{ .k = 4, .state_len = 8 });
    defer ring.deinit();
    const hooks = Hooks{ .snapshot = snapFake, .restore = restoreFake, .ctx = &fs };

    for (0..2) |li| {
        for (0..8) |i| fs.data[i] = @floatFromInt(li * 10 + i);
        _ = try ring.push(hooks, li, 2);
    }
    // El más reciente es la posición local 1 (slot 0).
    fs.data[0] = -1;
    _ = try ring.restoreAt(hooks, 1, 2);
    try std.testing.expectEqual(@as(f32, 10.0), fs.data[0]);
}

test "SsmRollback: posiciones fuera del ring → RingOverflow en push, false en restoreAt" {
    var fs = FakeState{ .data = undefined };
    var ring = try SsmRollback.init(test_allocator, .{ .k = 2, .state_len = 8 });
    defer ring.deinit();
    const hooks = Hooks{ .snapshot = snapFake, .restore = restoreFake, .ctx = &fs };

    // Ronda de 5 tokens, k=2: solo sobreviven las 2 posiciones más recientes.
    // push(pos 0..2) desborda el ring (dist >= 2) → error.RingOverflow.
    // Solo las posiciones 3 y 4 (dist 1 y 0) caben en el slot 1 y 0.
    for (0..5) |li| {
        @memset(fs.data[0..8], @floatFromInt(li));
        const r = ring.push(hooks, li, 5);
        if (li <= 2) {
            try std.testing.expectError(error.RingOverflow, r);
        } else {
            _ = try r;
        }
    }
    // Los únicos slots escritos: pos 4 (slot 0) y pos 3 (slot 1). valid = 2.
    try std.testing.expectEqual(@as(usize, 2), ring.valid);

    // Restaurar a posición 0 (muy atrás, dist 4 ≥ k) → false.
    const ok = try ring.restoreAt(hooks, 0, 5);
    try std.testing.expect(!ok);
    // Restaurar a posición 4 (más reciente, slot 0) → true.
    const ok_recent = try ring.restoreAt(hooks, 4, 5);
    try std.testing.expect(ok_recent);
    // El estado de la posición 4 era 4.0 → restore debe dar 4.0.
    try std.testing.expectEqual(@as(f32, 4.0), fs.data[0]);
}
