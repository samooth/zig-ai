//! Lane-b2 P2.3: Restore transaccional con rollback atómico.
//!
//! API standalone: opera sobre bytes puros + hooks. El caller (test o
//! wrapper de integración) enchufa las funciones de I/O del slot pool.
//!
//! ## Modelo
//!
//! `prepare` consume un `checkpoint.Snapshot` y produce un `RestorePlan`
//! con la lista de operaciones a aplicar. `commit(plan, ctx)` aplica las
//! ops. Si falla, `rollback(prev, ctx)` revierte al estado pre-restore
//! usando un `checkpoint.SlotSnapshot` capturado antes del restore.
//!
//! ## Uso típico
//!
//! ```zig
//! const ctx = RestoreCtx{ .read_slot = ..., .write_slot = ... };
//! var snap_prev = checkpoint.captureSlots(...);
//! defer snap_prev.deinit(allocator);
//!
//! var plan = try restore.prepare(allocator, snapshot);
//! defer plan.deinit(allocator);
//!
//! restore.commit(plan, ctx) catch |err| {
//!     restore.rollback(snap_prev, ctx);
//!     return err;
//! };
//! ```

const std = @import("std");
const checkpoint = @import("checkpoint.zig");

/// Contexto de restore: callbacks para I/O de slots del pool.
pub const RestoreCtx = struct {
    /// Lee los bytes de un slot del pool. `pool_index` es el idx del slot
    /// (0..n_slots). Devuelve el slice que apunta a la memoria del slot
    /// (no copia; el llamante debe respetar la vida).
    read_slot: *const fn (ctx: *const anyopaque, pool_index: u32) []const u8,

    /// Escribe bytes en un slot del pool. El caller es responsable de
    /// garantizar que el slot tiene capacidad suficiente.
    write_slot: *const fn (ctx: *const anyopaque, pool_index: u32, data: []const u8) void,

    /// Marca un slot como ocupado/desocupado (bool true = ocupado).
    set_slot_occupied: *const fn (ctx: *const anyopaque, pool_index: u32, occupied: bool) void,

    /// Opaque user pointer pasado a los callbacks.
    user_ctx: *const anyopaque = undefined,
};

/// Plan de restore: lista de operaciones a aplicar.
pub const RestorePlan = struct {
    allocator: std.mem.Allocator,
    ops: []Op,
    snapshot: checkpoint.Snapshot,

    const Self = @This();

    pub const Op = union(enum) {
        /// Copia bytes de `snapshot.body_buf[offset..]` o `tail_buf` al
        /// slot `pool_index`. El caller decide vía `is_tail`.
        write_slot: WriteOp,
        /// Marca slot como ocupado/desocupado.
        set_occupied: struct {
            pool_index: u32,
            occupied: bool,
        },

        pub const WriteOp = struct {
            pool_index: u32,
            /// Offset dentro del buffer del snapshot (body o tail, según
            /// `is_tail`).
            offset: u64,
            size: u64,
            /// `true` = escribir desde `tail_buf`; `false` = desde `body_buf`.
            is_tail: bool,
        };
    };

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.ops);
        self.snapshot.deinit(self.allocator);
        self.* = undefined;
    }
};

/// Pre-condiciones para un restore: el snapshot debe corresponder a una
/// seq activa en el manager con dimensiones compatibles.
pub const PrepareError = error{
    /// El seq_id del snapshot no existe en el manager.
    UnknownSeqId,
    /// Las dimensiones (num_layers, num_kv_heads, head_dim) no coinciden.
    DimensionsMismatch,
    /// El snapshot tiene cola exacta pero el manager no.
    TailRequiredButDisabled,
    /// El snapshot no tiene cola exacta pero el manager sí.
    TailDisabledButRequired,
    /// Versión incompatible.
    UnsupportedVersion,
    /// Allocator falló.
    OutOfMemory,
};

/// Prepara un restore desde un snapshot leído.
pub fn prepare(allocator: std.mem.Allocator, snapshot: checkpoint.Snapshot) PrepareError!RestorePlan {
    if (snapshot.header.version != checkpoint.CHECKPOINT_VERSION_V3) {
        return error.UnsupportedVersion;
    }
    if (snapshot.header.body_size != snapshot.body_buf.len) {
        return error.DimensionsMismatch;
    }
    if (snapshot.header.tail_size != snapshot.tail_buf.len) {
        return error.DimensionsMismatch;
    }
    if (snapshot.header.manifest_size != snapshot.manifest_buf.len) {
        return error.DimensionsMismatch;
    }

    const n_entries = @as(usize, snapshot.header.num_layers) * @as(usize, snapshot.header.num_kv_heads);

    // Count ops first so we allocate exactly the right amount.
    var op_count: usize = 0;
    {
        var j: usize = 0;
        while (j < n_entries) : (j += 1) {
            const e = checkpoint.ManifestEntry.read(
                snapshot.manifest_buf[j * checkpoint.MANIFEST_ENTRY_SIZE ..][0..checkpoint.MANIFEST_ENTRY_SIZE],
            );
            if (e.k_valid) op_count += 1;
            if (e.v_valid) op_count += 1;
            if (snapshot.header.tail_tokens > 0) {
                if (e.k_valid) op_count += 1;
                if (e.v_valid) op_count += 1;
            }
        }
    }

    var ops = try allocator.alloc(RestorePlan.Op, op_count);
    var op_idx: usize = 0;
    var body_off: u64 = 0;
    var tail_off: u64 = 0;

    var i: usize = 0;
    while (i < n_entries) : (i += 1) {
        const e = checkpoint.ManifestEntry.read(
            snapshot.manifest_buf[i * checkpoint.MANIFEST_ENTRY_SIZE ..][0..checkpoint.MANIFEST_ENTRY_SIZE],
        );
        if (e.k_valid) {
            const slot_size: u64 = @as(u64, e.k_stride) * @as(u64, snapshot.header.current_len);
            ops[op_idx] = .{ .write_slot = .{
                .pool_index = e.k_slot,
                .offset = body_off,
                .size = slot_size,
                .is_tail = false,
            } };
            op_idx += 1;
            body_off += slot_size;
        }
        if (e.v_valid) {
            const slot_size: u64 = @as(u64, e.v_stride) * @as(u64, snapshot.header.current_len);
            ops[op_idx] = .{ .write_slot = .{
                .pool_index = e.v_slot,
                .offset = body_off,
                .size = slot_size,
                .is_tail = false,
            } };
            op_idx += 1;
            body_off += slot_size;
        }
        if (snapshot.header.tail_tokens > 0) {
            const slab_bytes: u64 = @as(u64, snapshot.header.exact_groups) *
                @as(u64, snapshot.header.head_dim) * @sizeOf(f16);
            if (e.k_valid) {
                ops[op_idx] = .{ .write_slot = .{
                    .pool_index = e.k_slot,
                    .offset = tail_off,
                    .size = slab_bytes,
                    .is_tail = true,
                } };
                op_idx += 1;
                tail_off += slab_bytes;
            }
            if (e.v_valid) {
                ops[op_idx] = .{ .write_slot = .{
                    .pool_index = e.v_slot,
                    .offset = tail_off,
                    .size = slab_bytes,
                    .is_tail = true,
                } };
                op_idx += 1;
                tail_off += slab_bytes;
            }
        }
    }

    return .{
        .allocator = allocator,
        .ops = ops,
        .snapshot = snapshot,
    };
}

/// Error del commit con índice de la op que falló.
pub const CommitError = error{
    /// El offset+tamaño del body_buf cae fuera del buffer.
    BodyOutOfRange,
    /// El offset+tamaño del tail_buf cae fuera del buffer.
    TailOutOfRange,
    /// El hook write_slot reportó overflow (data > slot_size).
    WriteFailed,
};

/// Aplica el plan. Devuelve `CommitError` con la razón.
pub fn commit(plan: RestorePlan, ctx: RestoreCtx) CommitError!void {
    var i: usize = 0;
    while (i < plan.ops.len) : (i += 1) {
        switch (plan.ops[i]) {
            .write_slot => |ws| {
                const buf = if (ws.is_tail) plan.snapshot.tail_buf else plan.snapshot.body_buf;
                if (ws.offset + ws.size > buf.len) {
                    return if (ws.is_tail) error.TailOutOfRange else error.BodyOutOfRange;
                }
                const data = buf[ws.offset..][0..ws.size];
                ctx.write_slot(ctx.user_ctx, ws.pool_index, data);
            },
            .set_occupied => |so| ctx.set_slot_occupied(ctx.user_ctx, so.pool_index, so.occupied),
        }
    }
}

/// Rollback: revierte el estado a un SlotSnapshot previo.
pub fn rollback(prev: *const checkpoint.SlotSnapshot, ctx: RestoreCtx) void {
    const n_entries = @as(usize, prev.num_layers) * @as(usize, prev.num_kv_heads);
    for (0..n_entries) |i| {
        const k_off = i * prev.block_bytes;
        if (prev.k_occupied[i] and k_off + prev.block_bytes <= prev.k_slots.len) {
            ctx.write_slot(ctx.user_ctx, prev.k_slot_idx[i], prev.k_slots[k_off..][0..prev.block_bytes]);
            ctx.set_slot_occupied(ctx.user_ctx, prev.k_slot_idx[i], true);
        }
        if (prev.v_occupied[i] and k_off + prev.block_bytes <= prev.v_slots.len) {
            ctx.write_slot(ctx.user_ctx, prev.v_slot_idx[i], prev.v_slots[k_off..][0..prev.block_bytes]);
            ctx.set_slot_occupied(ctx.user_ctx, prev.v_slot_idx[i], true);
        }
        if (prev.exact_slab_bytes > 0) {
            const e_off = i * prev.exact_slab_bytes;
            if (prev.k_is_exact[i] and e_off + prev.exact_slab_bytes <= prev.exact_k_slabs.len) {
                ctx.write_slot(ctx.user_ctx, prev.k_slot_idx[i], prev.exact_k_slabs[e_off..][0..prev.exact_slab_bytes]);
            }
            if (prev.v_is_exact[i] and e_off + prev.exact_slab_bytes <= prev.exact_v_slabs.len) {
                ctx.write_slot(ctx.user_ctx, prev.v_slot_idx[i], prev.exact_v_slabs[e_off..][0..prev.exact_slab_bytes]);
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

/// Simula un "pool" de slots con un buffer + una tabla de ocupación.
const MockPool = struct {
    slots: [16][]u8,
    occupied: [16]bool,
    slot_size: usize,
};

fn readSlotImpl(ctx_ptr: *const anyopaque, idx: u32) []const u8 {
    const pool: *const MockPool = @ptrCast(@alignCast(ctx_ptr));
    return pool.slots[idx];
}
fn writeSlotImpl(ctx_ptr: *const anyopaque, idx: u32, data: []const u8) void {
    const pool: *MockPool = @ptrCast(@alignCast(@constCast(ctx_ptr)));
    if (data.len > pool.slot_size) return;
    @memcpy(pool.slots[idx][0..data.len], data);
}
fn setSlotImpl(ctx_ptr: *const anyopaque, idx: u32, occ: bool) void {
    const pool: *MockPool = @ptrCast(@alignCast(@constCast(ctx_ptr)));
    pool.occupied[idx] = occ;
}

test "prepare rejects future version" {
    const a = testing.allocator;
    var h = checkpoint.CheckpointHeader{ .version = 99 };
    var buf: [checkpoint.CheckpointHeader.SIZE]u8 = undefined;
    try h.write(&buf);
    var snap = checkpoint.Snapshot{
        .header = h,
        .manifest_buf = try a.alloc(u8, 0),
        .body_buf = try a.alloc(u8, 0),
        .tail_buf = &[_]u8{},
    };
    defer snap.deinit(a);
    try testing.expectError(error.UnsupportedVersion, restore.prepare(a, snap));
}

test "prepare rejects body_size mismatch" {
    const a = testing.allocator;
    const h = checkpoint.CheckpointHeader{ .body_size = 100, .manifest_size = 0, .tail_size = 0 };
    var snap = checkpoint.Snapshot{
        .header = h,
        .manifest_buf = &[_]u8{},
        .body_buf = try a.alloc(u8, 50),
        .tail_buf = &[_]u8{},
    };
    defer snap.deinit(a);
    try testing.expectError(error.DimensionsMismatch, restore.prepare(a, snap));
}

test "commit applies all write_slot ops in order" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const h = checkpoint.CheckpointHeader{
        .num_layers = 1,
        .num_kv_heads = 2,
        .head_dim = 4,
        .current_len = 2,
        // 2 entries × 2 (K+V) × stride 4 × current_len 2 = 32 bytes
        .body_size = 2 * 2 * 4 * 2,
        .manifest_size = 28 * 2,
        .tail_size = 0,
    };
    var snap = checkpoint.Snapshot{
        .header = h,
        .manifest_buf = try a.alloc(u8, h.manifest_size),
        .body_buf = try a.alloc(u8, h.body_size),
        .tail_buf = &[_]u8{},
    };

    const e0 = checkpoint.ManifestEntry{
        .k_format = .q4_0,
        .v_format = .q8_0,
        .k_stride = 4,
        .v_stride = 4,
        .k_slot = 0,
        .v_slot = 1,
        .k_valid = true,
        .v_valid = true,
    };
    e0.write(snap.manifest_buf[0..28]);
    var e1 = e0;
    e1.k_slot = 2;
    e1.v_slot = 3;
    e1.write(snap.manifest_buf[28..56]);

    for (snap.body_buf, 0..) |*b, i| b.* = @intCast(((i + 1) * 10) % 256);

    var pool = MockPool{
        .slots = undefined,
        .occupied = [_]bool{ false } ** 16,
        .slot_size = 8,
    };
    for (&pool.slots) |*s| s.* = try a.alloc(u8, 8);

    const ctx = RestoreCtx{
        .read_slot = readSlotImpl,
        .write_slot = writeSlotImpl,
        .set_slot_occupied = setSlotImpl,
        .user_ctx = @ptrCast(&pool),
    };

    var plan = try restore.prepare(a, snap);
    defer plan.deinit();
    try restore.commit(plan, ctx);

    try testing.expectEqualSlices(u8, snap.body_buf[0..8], pool.slots[0][0..8]);
    try testing.expectEqualSlices(u8, snap.body_buf[8..16], pool.slots[1][0..8]);
    try testing.expectEqualSlices(u8, snap.body_buf[16..24], pool.slots[2][0..8]);
    try testing.expectEqualSlices(u8, snap.body_buf[24..32], pool.slots[3][0..8]);
}

test "commit then rollback restores prior state" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const h = checkpoint.CheckpointHeader{
        .num_layers = 1,
        .num_kv_heads = 1,
        .head_dim = 4,
        .current_len = 1,
        .body_size = 8,
        .manifest_size = 28,
        .tail_size = 0,
    };
    var snap = checkpoint.Snapshot{
        .header = h,
        .manifest_buf = try a.alloc(u8, h.manifest_size),
        .body_buf = try a.alloc(u8, h.body_size),
        .tail_buf = &[_]u8{},
    };

    const e = checkpoint.ManifestEntry{
        .k_format = .q4_0,
        .v_format = .q8_0,
        .k_stride = 4,
        .v_stride = 4,
        .k_slot = 0,
        .v_slot = 1,
        .k_valid = true,
        .v_valid = true,
    };
    e.write(snap.manifest_buf[0..28]);
    snap.body_buf[0..4].* = .{ 1, 2, 3, 4 };
    snap.body_buf[4..8].* = .{ 5, 6, 7, 8 };

    var pool = MockPool{
        .slots = undefined,
        .occupied = [_]bool{ false } ** 16,
        .slot_size = 4,
    };
    for (&pool.slots) |*s| s.* = try a.alloc(u8, 4);

    pool.slots[0][0..4].* = .{ 10, 20, 30, 40 };
    pool.slots[1][0..4].* = .{ 50, 60, 70, 80 };

    var prev = checkpoint.SlotSnapshot{
        .k_slots = try a.alloc(u8, 4),
        .v_slots = try a.alloc(u8, 4),
        .k_occupied = try a.alloc(bool, 1),
        .v_occupied = try a.alloc(bool, 1),
        .k_slot_idx = try a.alloc(u32, 1),
        .v_slot_idx = try a.alloc(u32, 1),
        .k_is_exact = try a.alloc(bool, 1),
        .v_is_exact = try a.alloc(bool, 1),
        .exact_k_slabs = &[_]u8{},
        .exact_v_slabs = &[_]u8{},
        .num_layers = 1,
        .num_kv_heads = 1,
        .block_bytes = 4,
        .exact_slab_bytes = 0,
    };
    @memcpy(prev.k_slots, pool.slots[0][0..4]);
    @memcpy(prev.v_slots, pool.slots[1][0..4]);
    prev.k_occupied[0] = true;
    prev.v_occupied[0] = true;
    prev.k_slot_idx[0] = 0;
    prev.v_slot_idx[0] = 1;
    prev.k_is_exact[0] = false;
    prev.v_is_exact[0] = false;

    const ctx = RestoreCtx{
        .read_slot = readSlotImpl,
        .write_slot = writeSlotImpl,
        .set_slot_occupied = setSlotImpl,
        .user_ctx = @ptrCast(&pool),
    };

    var plan = try restore.prepare(a, snap);
    defer plan.deinit();

    try restore.commit(plan, ctx);
    try testing.expectEqualSlices(u8, snap.body_buf[0..4], pool.slots[0][0..4]);
    try testing.expectEqualSlices(u8, snap.body_buf[4..8], pool.slots[1][0..4]);

    restore.rollback(&prev, ctx);
    try testing.expectEqualSlices(u8, prev.k_slots, pool.slots[0][0..4]);
    try testing.expectEqualSlices(u8, prev.v_slots, pool.slots[1][0..4]);
}

/// Helper module alias (Zig exige `restore` antes de los tests).
const restore = @This();