//! Lane-b2 P2.3: Bridge simplificado entre `KVCacheManager` y checkpoint/restore.
//!
//! Funciones utilitarias para restaurar slots del pool desde un
//! `SlotSnapshot`. CPU-only.

const std = @import("std");
const checkpoint = @import("checkpoint.zig");

/// Restaura los índices de slot de un `SlotSnapshot` a arrays planos
/// `[num_layers * num_kv_heads]` de u32.
pub fn applyRollbackToSlots(
    prev: *const checkpoint.SlotSnapshot,
    dst_k: []u32,
    dst_v: []u32,
) void {
    const n: usize = prev.num_layers * prev.num_kv_heads;
    for (0..n) |i| {
        dst_k[i] = if (prev.k_occupied[i]) prev.k_slot_idx[i] else std.math.maxInt(u32);
        dst_v[i] = if (prev.v_occupied[i]) prev.v_slot_idx[i] else std.math.maxInt(u32);
    }
}