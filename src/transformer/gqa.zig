const std = @import("std");
const Tensor = @import("core").Tensor;

/// GQA (Grouped Query Attention) — Broadcast de KV heads sin copia
/// 
/// En lugar de expandir físicamente K/V de [batch, num_kv_heads, seq, head_dim]
/// a [batch, num_heads, seq, head_dim] con memcpy, usamos índices lógicos.
/// 
/// group_size = num_heads / num_kv_heads
/// Cada KV-head sirve a `group_size` Q-heads consecutivas.

pub const GQAIndices = struct {
    num_heads: usize,
    num_kv_heads: usize,
    group_size: usize,

    pub fn init(num_heads: usize, num_kv_heads: usize) GQAIndices {
        std.debug.assert(num_heads >= num_kv_heads);
        std.debug.assert(num_heads % num_kv_heads == 0);
        return .{
            .num_heads = num_heads,
            .num_kv_heads = num_kv_heads,
            .group_size = num_heads / num_kv_heads,
        };
    }

    /// Dado un índice de Q-head, devuelve el índice de KV-head correspondiente
    pub fn qHeadToKvHead(self: GQAIndices, q_head: usize) usize {
        return q_head / self.group_size;
    }

    /// Dado un índice de KV-head, devuelve el primer Q-head que lo usa
    pub fn kvHeadToFirstQHead(self: GQAIndices, kv_head: usize) usize {
        return kv_head * self.group_size;
    }
};

/// Acceso indirecto a K/V para FlashAttention con GQA
/// En lugar de copiar, el kernel FA accede a K/V usando qHeadToKvHead()
/// Esta función prepara un índice de lookup para uso en kernels o loops
pub fn buildGQAIndex(
    allocator: std.mem.Allocator,
    num_heads: usize,
    num_kv_heads: usize,
) ![]usize {
    const indices = try allocator.alloc(usize, num_heads);
    const gqa = GQAIndices.init(num_heads, num_kv_heads);
    for (0..num_heads) |h| {
        indices[h] = gqa.qHeadToKvHead(h);
    }
    return indices;
}

/// Para CPU/validación: expandir K/V físicamente (fallback cuando el kernel no soporta GQA)
/// NOTA: Preferir acceso indirecto en producción. Esta función es para tests/validación.
pub fn expandGqaFallback(
    allocator: std.mem.Allocator,
    src: Tensor(f16),          // [batch, num_kv_heads, seq_len, head_dim]
    num_q_heads: usize,
) !Tensor(f16) {
    const batch_size = src.shape[0];
    const num_kv_heads = src.shape[1];
    const seq_len = src.shape[2];
    const head_dim = src.shape[3];
    const group_size = num_q_heads / num_kv_heads;

    var dst = try Tensor(f16).alloc(allocator, &.{ batch_size, num_q_heads, seq_len, head_dim });

    for (0..batch_size) |b| {
        for (0..num_kv_heads) |kv_h| {
            for (0..group_size) |g| {
                const q_h = kv_h * group_size + g;
                for (0..seq_len) |s| {
                    const src_offset = ((b * num_kv_heads + kv_h) * seq_len + s) * head_dim;
                    const dst_offset = ((b * num_q_heads + q_h) * seq_len + s) * head_dim;
                    @memcpy(dst.data[dst_offset..dst_offset + head_dim], 
                            src.data[src_offset..src_offset + head_dim]);
                }
            }
        }
    }
    return dst;
}

// ─── Tests ───

test "gqa indices" {
    const gqa = GQAIndices.init(32, 8);
    try std.testing.expectEqual(@as(usize, 4), gqa.group_size);
    try std.testing.expectEqual(@as(usize, 0), gqa.qHeadToKvHead(0));
    try std.testing.expectEqual(@as(usize, 0), gqa.qHeadToKvHead(3));
    try std.testing.expectEqual(@as(usize, 1), gqa.qHeadToKvHead(4));
    try std.testing.expectEqual(@as(usize, 7), gqa.qHeadToKvHead(31));
}

test "build gqa index" {
    const allocator = std.testing.allocator;
    const idx = try buildGQAIndex(allocator, 8, 2);
    defer allocator.free(idx);
    try std.testing.expectEqual(@as(usize, 0), idx[0]);
    try std.testing.expectEqual(@as(usize, 0), idx[1]);
    try std.testing.expectEqual(@as(usize, 0), idx[2]);
    try std.testing.expectEqual(@as(usize, 0), idx[3]);
    try std.testing.expectEqual(@as(usize, 1), idx[4]);
    try std.testing.expectEqual(@as(usize, 1), idx[7]);
}
