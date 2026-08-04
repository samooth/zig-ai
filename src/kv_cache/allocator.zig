//! Pool allocator especializado para bloques de KV-cache
//! Gestiona memoria contigua con estrategias: bump, free-list, LRU eviction

const std = @import("std");
const qt = @import("quant_types.zig");
const KVBlockDescriptor = qt.KVBlockDescriptor;
const QuantFormat = qt.QuantFormat;
const CacheSlot = qt.CacheSlot;

/// Estrategia de asignación de bloques
pub const AllocStrategy = enum {
    /// Asignación secuencial simple (bump pointer)
    bump,
    /// Reutilizar bloques liberados (free-list)
    free_list,
    /// Evicción LRU cuando se agota el espacio
    lru_evict,
};

/// Pool de memoria para KV-cache
pub const KVPoolAllocator = struct {
    allocator: std.mem.Allocator,
    /// Buffer contiguo de memoria
    buffer: []u8,
    /// Tamaño total del buffer
    capacity: usize,
    /// Offset de bump pointer
    bump_offset: usize,
    /// Free-list de offsets liberados
    free_list: std.ArrayList(usize),
    /// Slots de cache activos
    slots: std.ArrayList(CacheSlot),
    /// Mapa de índice de slot -> descriptor
    slot_map: std.AutoHashMap(u32, KVBlockDescriptor),
    /// Estrategia actual
    strategy: AllocStrategy,
    /// Contador de accesos para LRU
    access_counter: u64,
    /// Callback de evicción (slot_idx, descriptor)
    on_evict: ?*const fn (u32, KVBlockDescriptor) void,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        capacity_bytes: usize,
        strategy: AllocStrategy,
    ) !Self {
        const buffer = try allocator.alloc(u8, capacity_bytes);
        errdefer allocator.free(buffer);

        var free_list: std.ArrayList(usize) = .empty;
        errdefer free_list.deinit(allocator);

        var slots: std.ArrayList(CacheSlot) = .empty;
        errdefer slots.deinit(allocator);

        var slot_map = std.AutoHashMap(u32, KVBlockDescriptor).init(allocator);
        errdefer slot_map.deinit();

        return .{
            .allocator = allocator,
            .buffer = buffer,
            .capacity = capacity_bytes,
            .bump_offset = 0,
            .free_list = free_list,
            .slots = slots,
            .slot_map = slot_map,
            .strategy = strategy,
            .access_counter = 0,
            .on_evict = null,
        };
    }

    pub fn deinit(self: *Self) void {
        self.free_list.deinit(self.allocator);
        self.slots.deinit(self.allocator);
        self.slot_map.deinit();
        self.allocator.free(self.buffer);
    }

    /// Asigna un bloque contiguo de memoria
    pub fn allocBlock(
        self: *Self,
        layer_idx: u32,
        head_idx: u32,
        seq_start: u32,
        seq_len: u32,
        head_dim: u32,
        format: QuantFormat,
    ) !*CacheSlot {
        const num_elements = @as(usize, seq_len) * @as(usize, head_dim);
        const block_size = format.defaultBlockSize();
        const num_blocks = (num_elements + block_size - 1) / block_size;
        const byte_size = num_blocks * format.bytesPerBlock();

        // Intentar asignar
        const offset = try self.findSpace(byte_size);

        const slot_idx = @as(u32, @intCast(self.slots.items.len));
        const descriptor = KVBlockDescriptor{
            .layer_idx = layer_idx,
            .head_idx = head_idx,
            .seq_start = seq_start,
            .seq_len = seq_len,
            .head_dim = head_dim,
            .format = format,
            .byte_offset = offset,
            .byte_size = byte_size,
        };

        const slot = CacheSlot{
            .idx = slot_idx,
            .occupied = true,
            .ref_count = 1,
            .last_access = self.nextAccessCounter(),
            .descriptor = descriptor,
        };

        try self.slots.append(self.allocator, slot);
        try self.slot_map.put(slot_idx, descriptor);

        return &self.slots.items[slot_idx];
    }

    /// Libera un slot (decrementa ref_count, libera si llega a 0)
    pub fn freeSlot(self: *Self, slot_idx: u32) void {
        const slot = &self.slots.items[slot_idx];
        if (slot.ref_count > 0) {
            slot.ref_count -= 1;
            if (slot.ref_count == 0) {
                slot.occupied = false;
                if (self.strategy == .free_list or self.strategy == .lru_evict) {
                    self.free_list.append(self.allocator, slot.descriptor.byte_offset) catch {};
                }
            }
        }
    }

    /// Obtiene un puntero al buffer de un slot
    pub fn getBuffer(self: *Self, slot_idx: u32) ?[]u8 {
        const slot = &self.slots.items[slot_idx];
        if (!slot.occupied) return null;
        slot.last_access = self.nextAccessCounter();
        const desc = slot.descriptor;
        return self.buffer[desc.byte_offset .. desc.byte_offset + desc.byte_size];
    }

    /// Compacta la memoria (defragmentación)
    pub fn compact(self: *Self) !void {
        if (self.strategy != .lru_evict) return;

        var new_offset: usize = 0;
        var new_slots: std.ArrayList(CacheSlot) = .empty;
        defer new_slots.deinit(self.allocator);

        // Ordenar slots por offset
        const SortCtx = struct {
            slots: []CacheSlot,
            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                return ctx.slots[a].descriptor.byte_offset < ctx.slots[b].descriptor.byte_offset;
            }
        };

        var indices = try self.allocator.alloc(usize, self.slots.items.len);
        defer self.allocator.free(indices);
        for (0..indices.len) |i| indices[i] = i;

        std.sort.insertion(usize, indices, SortCtx{ .slots = self.slots.items }, SortCtx.lessThan);

        for (indices) |i| {
            var slot = self.slots.items[i];
            if (!slot.occupied) continue;

            const desc = slot.descriptor;
            if (desc.byte_offset != new_offset) {
                // Mover datos
                const src = self.buffer[desc.byte_offset .. desc.byte_offset + desc.byte_size];
                const dst = self.buffer[new_offset .. new_offset + desc.byte_size];
                @memcpy(dst, src);
                slot.descriptor.byte_offset = new_offset;
            }
            new_offset += desc.byte_size;
            try new_slots.append(self.allocator, slot);
        }

        self.bump_offset = new_offset;
        self.slots.clearRetainingCapacity();
        try self.slots.appendSlice(self.allocator, new_slots.items);
        self.free_list.clearRetainingCapacity();
    }

    /// Porcentaje de uso
    pub fn usagePercent(self: *Self) f32 {
        var used: usize = 0;
        for (self.slots.items) |slot| {
            if (slot.occupied) used += slot.descriptor.byte_size;
        }
        return (@as(f32, @floatFromInt(used)) / @as(f32, @floatFromInt(self.capacity))) * 100.0;
    }

    // ─── Internos ───

    fn findSpace(self: *Self, size: usize) !usize {
        // 1. Intentar free-list
        if (self.strategy == .free_list or self.strategy == .lru_evict) {
            var i: usize = 0;
            while (i < self.free_list.items.len) : (i += 1) {
                const offset = self.free_list.items[i];
                // Verificar si hay espacio contiguo (simplificado: asumimos que free-list
                // guarda bloques del tamaño exacto o mayor)
                if (self.canFit(offset, size)) {
                    _ = self.free_list.orderedRemove(i);
                    return offset;
                }
            }
        }

        // 2. Bump pointer
        if (self.bump_offset + size <= self.capacity) {
            const offset = self.bump_offset;
            self.bump_offset += size;
            return offset;
        }

        // 3. LRU eviction
        if (self.strategy == .lru_evict) {
            return try self.evictAndAlloc(size);
        }

        return error.OutOfMemory;
    }

    fn canFit(self: *Self, offset: usize, size: usize) bool {
        // Verificar que no solape con slots ocupados
        const end = offset + size;
        if (end > self.capacity) return false;
        for (self.slots.items) |slot| {
            if (!slot.occupied) continue;
            const s_start = slot.descriptor.byte_offset;
            const s_end = s_start + slot.descriptor.byte_size;
            if (offset < s_end and end > s_start) return false;
        }
        return true;
    }

    fn evictAndAlloc(self: *Self, size: usize) !usize {
        // Encontrar slot LRU con ref_count == 0
        var lru_idx: ?u32 = null;
        var lru_time: u64 = std.math.maxInt(u64);

        for (self.slots.items, 0..) |slot, i| {
            if (!slot.occupied or slot.ref_count > 0) continue;
            if (slot.last_access < lru_time) {
                lru_time = slot.last_access;
                lru_idx = @as(u32, @intCast(i));
            }
        }

        if (lru_idx) |idx| {
            const slot = &self.slots.items[idx];
            if (self.on_evict) |cb| {
                cb(idx, slot.descriptor);
            }
            const offset = slot.descriptor.byte_offset;
            slot.occupied = false;

            if (slot.descriptor.byte_size >= size) {
                // Reutilizar este bloque (puede sobrar espacio)
                return offset;
            }
        }

        // Si no hay suficiente, intentar compactar
        try self.compact();

        if (self.bump_offset + size <= self.capacity) {
            const offset = self.bump_offset;
            self.bump_offset += size;
            return offset;
        }

        return error.OutOfMemory;
    }

    fn nextAccessCounter(self: *Self) u64 {
        self.access_counter += 1;
        return self.access_counter;
    }
};
