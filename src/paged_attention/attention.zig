const std = @import("std");
const BlockTable = @import("block_table.zig").BlockTable;
const BlockAllocator = @import("allocator.zig").BlockAllocator;
const PagedConfig = @import("root.zig").PagedConfig;

pub const PagedAttention = struct {
    config: PagedConfig,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(gpa: std.mem.Allocator, config: PagedConfig) Self {
        return .{ .allocator = gpa, .config = config };
    }

    pub fn decode(
        self: *const Self,
        query: []const f32,
        out: []f32,
        block_table: *const BlockTable,
        block_alloc: *BlockAllocator,
    ) !void {
        const num_q_heads = self.config.num_q_heads;
        const num_kv_heads = self.config.num_kv_heads;
        const head_dim = self.config.head_dim;
        const q_per_kv = num_q_heads / num_kv_heads;
        const block_size = self.config.block_size;
        const seq_len = block_table.num_tokens;

        std.debug.assert(query.len == num_q_heads * head_dim);
        std.debug.assert(out.len == num_q_heads * head_dim);
        @memset(out, 0);

        var max_scores = try self.allocator.alloc(f32, num_q_heads);
        defer self.allocator.free(max_scores);
        @memset(max_scores, -std.math.inf(f32));

        var exp_sums = try self.allocator.alloc(f32, num_q_heads);
        defer self.allocator.free(exp_sums);
        @memset(exp_sums, 0);

        var acc = try self.allocator.alloc(f32, num_q_heads * head_dim);
        defer self.allocator.free(acc);
        @memset(acc, 0);

        const bytes_per_elem = block_alloc.bytes_per_elem;
        const kv_stride_block = block_size * num_kv_heads * head_dim * bytes_per_elem;

        const num_blocks = block_table.numBlocks();
        for (0..num_blocks) |block_idx| {
            const phys_id = block_table.getPhysical(block_idx) orelse continue;
            const block = &block_alloc.blocks[phys_id];
            const block_data = block_alloc.memory_pool[phys_id * block_alloc.block_bytes ..];

            const tokens_in_block = @min(block.num_tokens, block_size);
            const block_start_token = block_idx * block_size;

            for (0..tokens_in_block) |t| {
                const token_pos = block_start_token + t;
                if (token_pos >= seq_len) break;

                const kv_offset = t * num_kv_heads * head_dim * bytes_per_elem;

                for (0..num_q_heads) |qh| {
                    const kv_h = qh / q_per_kv;
                    var score: f32 = 0;
                    const q_base = qh * head_dim;

                    for (0..head_dim) |d| {
                        const k_idx = kv_offset + kv_h * head_dim * bytes_per_elem + d * bytes_per_elem;
                        const k_val = loadF16(block_data, k_idx);
                        score += query[q_base + d] * k_val;
                    }
                    score /= @sqrt(@as(f32, @floatFromInt(head_dim)));

                    const prev_max = max_scores[qh];
                    if (score > prev_max) {
                        const scale = @exp(prev_max - score);
                        exp_sums[qh] *= scale;
                        max_scores[qh] = score;
                    }
                    const exp_score = @exp(score - max_scores[qh]);
                    exp_sums[qh] += exp_score;

                    for (0..head_dim) |d| {
                        const v_idx = kv_offset + kv_h * head_dim * bytes_per_elem + d * bytes_per_elem + kv_stride_block;
                        const v_val = loadF16(block_data, v_idx);
                        acc[q_base + d] += v_val * exp_score;
                    }
                }
            }
        }

        for (0..num_q_heads) |qh| {
            const sum = exp_sums[qh];
            const base = qh * head_dim;
            if (sum > 0) {
                for (0..head_dim) |d| {
                    out[base + d] = acc[base + d] / sum;
                }
            }
        }
    }

    pub fn decodeBatch(
        self: *const Self,
        queries: []const f32,
        outs: []f32,
        block_tables: []const *const BlockTable,
        block_alloc: *BlockAllocator,
    ) !void {
        const batch_size = block_tables.len;
        const q_stride = self.config.num_q_heads * self.config.head_dim;

        for (0..batch_size) |b| {
            const q_off = b * q_stride;
            const o_off = b * q_stride;
            try self.decode(
                queries[q_off..][0..q_stride],
                outs[o_off..][0..q_stride],
                block_tables[b],
                block_alloc,
            );
        }
    }

    pub fn prefill(
        self: *const Self,
        queries: []const f32,
        outs: []f32,
        block_table: *const BlockTable,
        block_alloc: *BlockAllocator,
        seq_len: usize,
    ) !void {
        const q_stride = self.config.num_q_heads * self.config.head_dim;
        for (0..seq_len) |i| {
            try self.decode(
                queries[i * q_stride ..][0..q_stride],
                outs[i * q_stride ..][0..q_stride],
                block_table,
                block_alloc,
            );
        }
    }

    fn loadF16(data: []const u8, offset: usize) f32 {
        const bits: u16 = @as(u16, data[offset]) | (@as(u16, data[offset + 1]) << 8);
        const sign: u32 = @as(u32, (bits >> 15) & 1);
        const exp: i32 = @as(i32, @intCast((bits >> 10) & 0x1F)) - 15;
        const mant: u32 = @as(u32, bits & 0x3FF);
        if (exp == -15 and mant == 0) return 0;
        if (exp == 16) return if (sign == 1) -std.math.inf(f32) else std.math.inf(f32);
        const fexp: f32 = @exp2(@as(f32, @floatFromInt(exp)));
        const fmant: f32 = 1.0 + @as(f32, @floatFromInt(mant)) / 1024.0;
        const val = fexp * fmant;
        return if (sign == 1) -val else val;
    }
};
