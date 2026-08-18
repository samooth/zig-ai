//! Pool de bloques GPU con memoria paginada (CUDA VMM: cuMemAddressReserve /
//! cuMemCreate / cuMemMap). Reserva un rango de direcciones virtuales para
//! `num_blocks * block_bytes` y hace commit físico por bloque de forma
//! perezosa: `stageBlock` crea un handle (cuMemCreate), lo mapea (cuMemMap +
//! cuMemSetAccess) y copia H2D; `evictBlock` copia D2H, desmapea y libera el
//! handle. Solo los bloques residentes consumen memoria física del dispositivo.
//! Requiere `block_bytes` alineado a la granularidad mínima del dispositivo
//! para que el layout `phys * block_bytes` del kernel coincida con los
//! mappings VMM. Si no encaja, usar `GpuBlockPool` contiguo.
const std = @import("std");
const debug = @import("debug");
const cudaz = @import("cudaz");
const BlockAllocator = @import("allocator.zig").BlockAllocator;
const BlockTable = @import("block_table.zig").BlockTable;

pub const PagedGpuBlockPool = struct {
    allocator: std.mem.Allocator,
    num_blocks: usize,
    block_bytes: usize,
    granule: usize,
    vaddr: cudaz.CUdeviceptr,
    handles: []cudaz.CUmemGenericAllocationHandle,
    resident: []bool,
    dirty: []bool,

    const Self = @This();

    /// Retorna la granularidad mínima de alocación del dispositivo, o null si
    /// el driver no soporta VMM.
    pub fn getGranule() !usize {
        try cudaz.ensureCurrent();
        var prop = cudaz.CUmemAllocationProp{
            .type = .PINNED,
            .requestedHandleTypes = .NONE,
            .location = .{ .type = .DEVICE, .id = 0 },
        };
        var granule: usize = 0;
        try cudaz.cuMemGetAllocationGranularity(&granule, &prop, cudaz.CU_MEM_ALLOCATION_GRANULARITY_MINIMUM);
        return if (granule == 0) 64 * 1024 else granule;
    }

    pub fn init(gpa: std.mem.Allocator, num_blocks: usize, block_bytes: usize) !Self {
        try cudaz.ensureCurrent();
        const granule = try getGranule();
        if (block_bytes % granule != 0) return error.BlockBytesNotGranuleAligned;

        const total = num_blocks * block_bytes;

        var vaddr: cudaz.CUdeviceptr = 0;
        try cudaz.cuMemAddressReserve(&vaddr, total, granule, 0, 0);
        errdefer cudaz.cuMemAddressFree(vaddr, total);

        const handles = try gpa.alloc(cudaz.CUmemGenericAllocationHandle, num_blocks);
        errdefer gpa.free(handles);
        @memset(handles, 0);

        const resident = try gpa.alloc(bool, num_blocks);
        errdefer gpa.free(resident);
        @memset(resident, false);

        const dirty = try gpa.alloc(bool, num_blocks);
        errdefer gpa.free(dirty);
        @memset(dirty, false);

        return .{
            .allocator = gpa,
            .num_blocks = num_blocks,
            .block_bytes = block_bytes,
            .granule = granule,
            .vaddr = vaddr,
            .handles = handles,
            .resident = resident,
            .dirty = dirty,
        };
    }

    pub fn deinit(self: *Self) void {
        for (0..self.num_blocks) |phys| {
            if (self.resident[phys]) {
                cudaz.cuMemUnmap(self.blockAddr(phys), self.block_bytes);
                cudaz.cuMemRelease(self.handles[phys]);
            }
        }
        cudaz.cuMemAddressFree(self.vaddr, self.num_blocks * self.block_bytes);
        self.allocator.free(self.handles);
        self.allocator.free(self.resident);
        self.allocator.free(self.dirty);
    }

    pub fn markDirty(self: *Self, phys_id: usize) void {
        if (phys_id >= self.num_blocks) return;
        self.dirty[phys_id] = true;
    }

    fn blockAddr(self: *const Self, phys_id: usize) cudaz.CUdeviceptr {
        return self.vaddr + phys_id * self.block_bytes;
    }

    pub fn stageBlock(self: *Self, block_alloc: *BlockAllocator, phys_id: usize) !void {
        debug.dbg.printLevel(.detail, "paged_gpu_pool: stageBlock phys_id={}\n", .{phys_id});
        if (phys_id >= self.num_blocks) return;
        if (self.resident[phys_id] and !self.dirty[phys_id]) return;
        if (!self.resident[phys_id]) {
            var prop = cudaz.CUmemAllocationProp{
                .type = .PINNED,
                .requestedHandleTypes = .NONE,
                .location = .{ .type = .DEVICE, .id = 0 },
            };
            var handle: cudaz.CUmemGenericAllocationHandle = 0;
            try cudaz.cuMemCreate(&handle, self.block_bytes, &prop, 0);
            errdefer cudaz.cuMemRelease(handle);
            try cudaz.cuMemMap(self.blockAddr(phys_id), self.block_bytes, 0, handle, 0);
            var access = cudaz.CUmemAccessDesc{
                .location = .{ .type = .DEVICE, .id = 0 },
                .flags = .PROT_READWRITE,
            };
            try cudaz.cuMemSetAccess(self.blockAddr(phys_id), self.block_bytes, &access, 1);
            self.handles[phys_id] = handle;
            self.resident[phys_id] = true;
        }
        const src = @intFromPtr(block_alloc.memory_pool.ptr) + phys_id * self.block_bytes;
        debug.dbg.printLevel(.detail, "paged_gpu_pool: cuMemcpyHtoD phys_id={} src={} bytes={}\n", .{phys_id, src, self.block_bytes});
        try cudaz.cuMemcpyHtoD(self.blockAddr(phys_id), src, self.block_bytes);
        debug.dbg.printLevel(.detail, "paged_gpu_pool: cuMemcpyHtoD done phys_id={}\n", .{phys_id});
        self.dirty[phys_id] = false;
    }

    pub fn evictBlock(self: *Self, block_alloc: *BlockAllocator, phys_id: usize) !void {
        if (phys_id >= self.num_blocks or !self.resident[phys_id]) return;
        const dst = @intFromPtr(block_alloc.memory_pool.ptr) + phys_id * self.block_bytes;
        try cudaz.cuMemcpyDtoH(dst, self.blockAddr(phys_id), self.block_bytes);
        cudaz.cuMemUnmap(self.blockAddr(phys_id), self.block_bytes);
        cudaz.cuMemRelease(self.handles[phys_id]);
        self.resident[phys_id] = false;
    }

    pub fn stageTable(self: *Self, block_alloc: *BlockAllocator, block_table: *const BlockTable) !void {
        for (0..block_table.numBlocks()) |i| {
            if (block_table.getPhysical(i)) |phys| try self.stageBlock(block_alloc, phys);
        }
    }

    /// Marca el bloque residente y lo mapea (VMM) sin copiar H2D: el KV se
    /// escribe por GPU (kvAppendF16) directamente sobre la memoria mapeada.
    pub fn ensureCommitted(self: *Self, phys_id: usize) !void {
        if (phys_id >= self.num_blocks) return;
        if (!self.resident[phys_id]) {
            var prop = cudaz.CUmemAllocationProp{
                .type = .PINNED,
                .requestedHandleTypes = .NONE,
                .location = .{ .type = .DEVICE, .id = 0 },
            };
            var handle: cudaz.CUmemGenericAllocationHandle = 0;
            try cudaz.cuMemCreate(&handle, self.block_bytes, &prop, 0);
            errdefer cudaz.cuMemRelease(handle);
            try cudaz.cuMemMap(self.blockAddr(phys_id), self.block_bytes, 0, handle, 0);
            var access = cudaz.CUmemAccessDesc{
                .location = .{ .type = .DEVICE, .id = 0 },
                .flags = .PROT_READWRITE,
            };
            try cudaz.cuMemSetAccess(self.blockAddr(phys_id), self.block_bytes, &access, 1);
            self.handles[phys_id] = handle;
            self.resident[phys_id] = true;
        }
        self.dirty[phys_id] = false;
    }

    /// Baja (async, stream-ordered) el bloque escrito por GPU al host pool.
    pub fn syncBlockToHost(self: *Self, block_alloc: *BlockAllocator, phys_id: usize, stream: cudaz.CUstream) !void {
        if (phys_id >= self.num_blocks or !self.resident[phys_id]) return;
        const dst = @intFromPtr(block_alloc.memory_pool.ptr) + phys_id * self.block_bytes;
        try cudaz.cuMemcpyDtoHAsync(dst, self.blockAddr(phys_id), self.block_bytes, stream);
    }

    pub fn evictBlocks(self: *Self, block_alloc: *BlockAllocator, phys_ids: []const usize) !void {
        for (phys_ids) |phys| {
            try self.evictBlock(block_alloc, phys);
        }
    }

    pub fn numResident(self: *const Self) usize {
        var n: usize = 0;
        for (self.resident) |r| {
            if (r) n += 1;
        }
        return n;
    }
};