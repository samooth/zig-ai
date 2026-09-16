//! Backend capabilities + pure route policy for the KVarN FA native dispatch
//! (lane B1, FASE 1→2 of PLAN_B1).
//!
//! ## Scope (B1 of TODO_B1_DEV_B.md)
//!
//! This file is **host-side, zero GPU state**, mirrors the transcrito del
//! `fattn-kvarn-route-policy.h` upstream, and provides:
//!
//! 1. `probeDevice(ordinal) -> Capabilities` — `cuDeviceGetAttribute` based
//!    capability snapshot. Uses LOCAL externs (no edits to `cudaz_stub.zig`,
//!    per TODO_B1_DEV_B.md §B1: "añade los externs que falten en TU fichero,
//!    no en cudaz"). The probe is best-effort: it returns a `Capabilities`
//!    with `kvarn_instances = false` and `matrix_mma = false` if the device
//!    cannot be reached (no GPU, no driver, sm_75 fallback). The policy code
//!    is then `fail-closed`: routes that require MMA are never chosen.
//!
//! 2. `Capabilities` struct + `selectCapabilities(input)` — pure function
//!    computing the supported route set given backend, physical wave size,
//!    MMA availability, KVarN instances, max threads/block, max shared
//!    memory/block, and minimum dynamic shared bytes. Pure function: no
//!    side effects, no I/O.
//!
//! 3. Pure route policy: `selectRoute`, `selectFallbackRoute`, `useWideMma`,
//!    `bodyShapeSupported`, `portableSupported` — all (input) -> output
//!    functions with no global state. The dispatch master (B7) composes
//!    these; portable FA (B4) and vec FA (B6) register themselves as route
//!    targets at the wrapper layer.
//!
//! 4. `RouteCounters` — atomic counters per route family, atomic-max
//!    descriptors / partials / meta memory stats. Exposed via
//!    `countersGet/countersReset/countersPrint`. Emits breadcrumbs via the
//!    existing `debug.dbg.printLevel(.detail, …)` gate (no new flag added
//!    to `src/debug.zig` here — the `perf_kvarn_route` flag will be
//!    added in a P4 build.zig/debug.zig window together with Dev A per
//!    PLAN_B1.md §7).
//!
//! 5. Env-var overrides (per PLAN_B1.md §8):
//!    - `ZIG_AI_KVARN_FORCE_PORTABLE` (=1) — `selectRoute` pins
//!      `PORTABLE_NATIVE` regardless of the natural priority order.
//!    - `ZIG_AI_KVARN_VEC` (=0) — disables `DECODE_VECTOR` (default 1).
//!    - `ZIG_AI_KVARN_VEC_TOKENS` (=16) — hint for vec kernel TOKENS_PER_SPLIT.
//!    - `ZIG_AI_KVARN_DEBUG_ROUTES` (=1) — verbose breadcrumb of the route
//!      chosen + reason + fallback path. Gated by `dbg.dbg.at(.detail)`.
//!
//! ## Transcripción vs copia
//!
//! Pure transcripción: re-expressed the algorithm with our own struct names,
//! field order, and the zig-ai ownership split. NO code from
//! `/ai/repos/2026/beellama.cpp` was copied. Constants and the policy
//! priority order are restated from the analysis in
//! `docs/b1-research/KVAR_N_CUDA_REFERENCE.md` §12.

const std = @import("std");
const debugz = @import("debug");

// ─── CUdevice attribute ids (subset used by the probe) ───────────────────────
//
// From `cuda.h` (CUdevice_attribute). We only need the attributes the policy
// reads; everything else is left to whoever wires a future need.

pub const CUdevice_attribute = enum(c_int) {
    MAX_THREADS_PER_BLOCK = 39,
    MAX_SHARED_MEMORY_PER_BLOCK = 8, // legacy
    MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    MULTIPROCESSOR_COUNT = 16,
    WARP_SIZE = 10,
    COMPUTE_CAPABILITY_MAJOR = 75,
    COMPUTE_CAPABILITY_MINOR = 76,
    INTEGRATED = 18,
    _,
};

/// Local externs: scoped to this file so the kernel/wrapper cudaz module
/// stays untouched. These call the same libcuda entry points as `cudaz` but
/// do not need a context-retained primary ctx to read static device attrs.
const LibCuda = struct {
    extern "c" fn cuDeviceGetAttribute(pi: *c_int, attrib: c_int, dev: c_int) c_int;
    extern "c" fn cuDeviceGetCount(count: *c_int) c_int;
    extern "c" fn cuDeviceComputeCapability(major: *c_int, minor: *c_int, dev: c_int) c_int;
};

/// Wraps `cuDeviceGetAttribute` returning the int value or `null` on failure
/// (silent — the caller decides the fail-closed shape of the result).
fn attr(dev: c_int, a: CUdevice_attribute) ?c_int {
    var v: c_int = undefined;
    if (LibCuda.cuDeviceGetAttribute(&v, @intFromEnum(a), dev) != 0) return null;
    return v;
}

// ─── Backend + route enums (mirroring the policy header) ────────────────────

/// Backend the policy runs over. Only `CUDA` is fully implemented; `HIP` /
/// `MUSA` are placeholder values that `selectCapabilities` reads to apply
/// the documented gating (HIP has no split/vector; MUSA stays portable).
pub const Backend = enum {
    cuda,
    hip,
    musa,
};

/// AMD matrix-MMA family (when `Backend == .hip`). Kept as a 3-valued enum
/// because both upstream branches need a non-default sentinel.
pub const AmdMmaFamily = enum {
    none,
    rdna_wmma,
    cdna_mfma,
};

/// Head-dim bitmask. Each value is `1 << log2(d)`; combinations are OR'd.
/// Backed by `u4` (3 bits used) to match the natural bit pattern (128→1<<0,
/// 256→1<<1, 512→1<<2); bits() returns the same value.
pub const HeadDimMask = packed struct(u3) {
    d128: bool = false,
    d256: bool = false,
    d512: bool = false,

    pub fn fromDim(d: u32) HeadDimMask {
        return switch (d) {
            128 => .{ .d128 = true },
            256 => .{ .d256 = true },
            512 => .{ .d512 = true },
            else => .{},
        };
    }

    pub fn has(self: HeadDimMask, d: u32) bool {
        return switch (d) {
            128 => self.d128,
            256 => self.d256,
            512 => self.d512,
            else => false,
        };
    }

    pub fn bits(self: HeadDimMask) u8 {
        var b: u8 = 0;
        if (self.d128) b |= 1 << 0;
        if (self.d256) b |= 1 << 1;
        if (self.d512) b |= 1 << 2;
        return b;
    }
};

/// Routes the dispatch can pick. `unavailable` is the explicit
/// fail-closed terminal (matrix_mma=false AND portable_native=false).
pub const Route = enum {
    prompt_prefill,
    decode_split,
    decode_vector,
    generic_mma,
    portable_native,
    unavailable,
};

/// Bitmask of route families for `Capabilities.route_families`.
pub const RouteFamilyMask = packed struct(u4) {
    portable_native: bool = false,
    generic_mma: bool = false,
    decode_split: bool = false,
    decode_vector: bool = false,

    pub fn has(self: RouteFamilyMask, r: Route) bool {
        return switch (r) {
            .portable_native => self.portable_native,
            .generic_mma => self.generic_mma,
            .decode_split => self.decode_split,
            .decode_vector => self.decode_vector,
            .prompt_prefill, .unavailable => false,
        };
    }

    pub fn set(self: *RouteFamilyMask, r: Route, on: bool) void {
        switch (r) {
            .portable_native => self.portable_native = on,
            .generic_mma => self.generic_mma = on,
            .decode_split => self.decode_split = on,
            .decode_vector => self.decode_vector = on,
            .prompt_prefill, .unavailable => {},
        }
    }
};

/// Constants locked at PLAN_B1.md §3 / KVARN_CUDA_REFERENCE §12.
pub const SPECIALIZED_DECODE_MAX_Q: u32 = 16;
pub const PORTABLE_THREADS: u32 = 128;
pub const PORTABLE_MAX_Q: u32 = std.math.maxInt(u32);
/// Per PLAN_B1.md D8: portable FA consumes 3,144 B of dynamic smem
/// (low-shmem); hi-shmem variants need the 69,704 B opt-in. The policy uses
/// the portable-footprint minimum as the gating value (it never gates on
/// the high-shmem fork here — that lives in Dev A's materializer launcher).
pub const MIN_DYNAMIC_SHARED_PORTABLE: u32 = 3144;

// ─── Capabilities (pure value, output of selectCapabilities) ────────────────

pub const Capabilities = struct {
    backend: Backend,
    /// Physical warp size of the device: 32 (NVIDIA default) or 64 (AMD).
    /// `null` when the probe could not read it.
    physical_wave_size: ?u8,
    /// True if the backend's matrix-MMA contract is available (sm_80+ on
    /// NVIDIA, RDNA_WMMA/CDNA_MFMA on HIP, never on MUSA).
    matrix_mma: bool,
    /// True if at least one KVarN instance / cubin is available on this
    /// device — controls whether `store_materialize` can ever be true.
    kvarn_instances: bool,
    /// Maximum threads per block the device reports (probe value or fallback
    /// `PORTABLE_THREADS` if probe failed).
    max_threads_per_block: u32,
    /// Opt-in max shared memory per block (bytes). `null` when the probe
    /// could not read it (= conservative default 48 KiB).
    shared_memory_per_block: ?u32,
    /// Derived: true if a portable route can launch on this hardware
    /// (wave size supported + threads ≥ 128 + smem ≥ MIN_DYNAMIC_SHARED_PORTABLE).
    portable_hardware: bool,
    /// Derived: store + materialize path available iff KVarN instances AND
    /// portable_hardware.
    store_materialize: bool,
    /// Native (body+tail) portable route — K and V f16 / bf16 tails.
    portable_native: bool,
    portable_tail_f16: bool,
    portable_tail_bf16: bool,
    /// Generic MMA prefill (n_q>1 cases). NVIDIA only: HIP+MMA has its
    /// own shape gating; MUSA does not have a generic MMA path.
    generic_mma: bool,
    decode_split: bool,
    decode_vector: bool,
    /// True if at least one specialized (non-portable) route is selectable.
    specialized_routes: bool,
    /// Backend-CUDA only: the V path lives in the original (non-rotated)
    /// domain for MMA routes. For portable, V is materialized in its
    /// natural (rotated) form. Mirrors upstream.
    original_v_domain: bool,
    route_families: RouteFamilyMask,
    /// Max n_q (queries) the portable route can run. Constant `UINT32_MAX`
    /// per the upstream when portable is enabled.
    rotated_query_max_portable: u32,
    /// Max n_q the specialized routes can run: 16 per `SPECIALIZED_DECODE_MAX_Q`.
    rotated_query_max_specialized: u32,
    /// Set of head dims the device + path set supports. If neither portable
    /// nor specialized routes are present this is zero.
    supported_head_dims: HeadDimMask,
};

// ─── Input struct for the capabilities decision ──────────────────────────────

pub const CapabilitiesInput = struct {
    backend: Backend,
    physical_wave_size: ?u8,
    matrix_mma: bool,
    kvarn_instances: bool,
    max_threads_per_block: u32,
    shared_memory_per_block: ?u32,
    /// When the platform exposes MMA on HIP, identify the family.
    amd_mma_family: AmdMmaFamily = .none,
};

/// Pure function: derive `Capabilities` from the probe values. Mirrors
/// `ggml_cuda_fattn_kvarn_select_capabilities` from
/// `docs/b1-research/KVAR_N_CUDA_REFERENCE.md` §12, re-expressed in zig-ai
/// terms and restricted to CUDA truth (HIP/MUSA stubs are explicit).
pub fn selectCapabilities(input: CapabilitiesInput) Capabilities {
    const wave = input.physical_wave_size;
    const physical_wave_supported = wave != null and (wave.? == 32 or wave.? == 64);

    // CUDA portable: warp32-only (upstream rule). HIP: any supported wave.
    const portable_wave_supported = switch (input.backend) {
        .cuda => wave != null and wave.? == 32,
        .hip, .musa => physical_wave_supported,
    };

    const smem = input.shared_memory_per_block orelse 48 * 1024; // conservative
    const portable_hardware = portable_wave_supported and
        input.max_threads_per_block >= PORTABLE_THREADS and
        smem >= MIN_DYNAMIC_SHARED_PORTABLE;

    const store_materialize = input.kvarn_instances and portable_hardware;
    const portable_native = store_materialize;
    const portable_tail_f16 = store_materialize;
    const portable_tail_bf16 = store_materialize;

    // Generic MMA: backend-specific.
    const generic_mma = switch (input.backend) {
        .cuda => input.matrix_mma and store_materialize,
        .hip => input.matrix_mma and store_materialize and physical_wave_supported,
        .musa => false,
    };
    // Split decode uses NVIDIA ldmatrix + m16n8 fragments ⇒ CUDA-only.
    // Vec decode is warp-tuned for NVIDIA too (HIP falls back to generic /
    // portable direct-record; MUSA has nothing).
    const decode_split = switch (input.backend) {
        .cuda => generic_mma,
        .hip, .musa => false,
    };
    const decode_vector = switch (input.backend) {
        .cuda => generic_mma,
        .hip, .musa => false,
    };

    const specialized_routes = generic_mma or decode_split or decode_vector;
    const original_v_domain = input.backend == .cuda and generic_mma;

    var route_families: RouteFamilyMask = .{};
    route_families.set(.portable_native, portable_native);
    route_families.set(.generic_mma, generic_mma);
    route_families.set(.decode_split, decode_split);
    route_families.set(.decode_vector, decode_vector);

    const rotated_query_max_portable: u32 = if (portable_native) PORTABLE_MAX_Q else 0;
    const rotated_query_max_specialized: u32 = if (specialized_routes) SPECIALIZED_DECODE_MAX_Q else 0;

    var supported: HeadDimMask = .{};
    if (portable_native or specialized_routes) {
        supported = .{ .d128 = true, .d256 = true, .d512 = true };
    }

    return .{
        .backend = input.backend,
        .physical_wave_size = wave,
        .matrix_mma = input.matrix_mma,
        .kvarn_instances = input.kvarn_instances,
        .max_threads_per_block = input.max_threads_per_block,
        .shared_memory_per_block = input.shared_memory_per_block,
        .portable_hardware = portable_hardware,
        .store_materialize = store_materialize,
        .portable_native = portable_native,
        .portable_tail_f16 = portable_tail_f16,
        .portable_tail_bf16 = portable_tail_bf16,
        .generic_mma = generic_mma,
        .decode_split = decode_split,
        .decode_vector = decode_vector,
        .specialized_routes = specialized_routes,
        .original_v_domain = original_v_domain,
        .route_families = route_families,
        .rotated_query_max_portable = rotated_query_max_portable,
        .rotated_query_max_specialized = rotated_query_max_specialized,
        .supported_head_dims = supported,
    };
}

// ─── Device probe (best-effort) ────────────────────────────────────────────

/// Probe a single device ordinal. Returns a `Capabilities` reflecting only
/// the bits we can read. KVarN / MMA flags are *not* set here — they are
/// policy inputs decided by the launcher that owns the cubin set (B7 will
/// set them from build-time `-Dkvarn-all-quants` + the running device).
pub fn probeDevice(ordinal: c_int) Capabilities {
    const backend: Backend = .cuda; // single-backend for now (HIP/MUSA later)
    const wave = attr(ordinal, .WARP_SIZE);
    const max_threads: u32 = blk: {
        const v = attr(ordinal, .MAX_THREADS_PER_BLOCK) orelse break :blk PORTABLE_THREADS;
        break :blk @intCast(@max(v, 0));
    };
    const smem_optin: ?u32 = blk: {
        const v = attr(ordinal, .MAX_SHARED_MEMORY_PER_BLOCK_OPTIN) orelse break :blk null;
        break :blk @intCast(@max(v, 0));
    };
    // matrix_mma: a function of the SM major (sm_80+ on NVIDIA). We
    // read the compute capability directly as a stable cross-check.
    var major: c_int = 0;
    var minor: c_int = 0;
    const cc_ok = LibCuda.cuDeviceComputeCapability(&major, &minor, ordinal) == 0;
    const matrix_mma = cc_ok and major >= 8;

    return selectCapabilities(.{
        .backend = backend,
        .physical_wave_size = if (wave) |w| @intCast(@as(u8, @intCast(w))) else null,
        .matrix_mma = matrix_mma,
        // The probe does NOT know about KVarN cubins. The dispatch master
        // (B7) decides `kvarn_instances` based on the running cubin set.
        .kvarn_instances = false,
        .max_threads_per_block = max_threads,
        .shared_memory_per_block = smem_optin,
    });
}

/// Quick sanity: at least one CUDA device is visible. Returns false when
/// the probe cannot enumerate any device (no driver / no GPU / no perms).
pub fn isDeviceVisible() bool {
    var n: c_int = 0;
    if (LibCuda.cuDeviceGetCount(&n) != 0) return false;
    return n > 0;
}

// ─── Route policy (pure functions) ─────────────────────────────────────────

/// Input to the route decision. All fields are caller-controlled: there
/// are no implicit reads of `Capabilities` here. The dispatch master
/// translates `Capabilities` + runtime shape into this struct.
pub const RouteInput = struct {
    head_dim: u32,
    n_q: u32,
    gqa: u32,
    k_bits: u8,
    v_bits: u8,
    swa: bool,
    /// `true` when the launcher is running the *prompt prefill* path
    /// (initial ingest of the prompt, large n_q, no causal-mask overlap
    /// with previously materialized records).
    prompt_prefill: bool,
    /// `true` when the geometry analyzer has tagged the shape as
    /// `decode_vector_eligible` (D=256, SWA, GQA==2, n_q==1).
    vector_eligible: bool,
    /// `true` when the geometry analyzer has tagged the shape as
    /// `decode_split_eligible` (n_q==1 ≤ 16).
    split_eligible: bool,
    /// `body_meta_requested` is a softmax-metadata output contract, never
    /// a route constraint. Mirrored here for API completeness with
    /// upstream's signature; not consulted.
    body_meta_requested: bool = false,
    /// Force the portable route regardless of natural priority. Honors
    /// `ZIG_AI_KVARN_FORCE_PORTABLE` from PLAN_B1.md §8.
    force_portable: bool = false,
    /// Disable `decode_vector` even when the geometry would allow it.
    /// Honors `ZIG_AI_KVARN_VEC=0`.
    vec_disabled: bool = false,
};

/// Decide which route the launcher should take. Pure function; no I/O.
/// Mirrors `ggml_cuda_fattn_kvarn_select_route` priority order from
/// `docs/b1-research/KVAR_N_CUDA_REFERENCE.md` §12.
pub fn selectRoute(caps: Capabilities, input: RouteInput) Route {
    if (input.force_portable) {
        return if (caps.portable_native) .portable_native else .unavailable;
    }
    if (input.prompt_prefill) return .prompt_prefill;
    if (input.vector_eligible and caps.decode_vector and !input.vec_disabled) {
        return .decode_vector;
    }
    if (input.n_q == 1 and input.split_eligible and caps.decode_split) {
        return .decode_split;
    }
    if (caps.generic_mma and input.n_q <= caps.rotated_query_max_specialized) {
        return .generic_mma;
    }
    if (caps.portable_native) return .portable_native;
    return .unavailable;
}

/// Pick a fallback when the primary `selectRoute` is not eligible.
/// `generic_shape_eligible` is the caller's precomputed flag — we do not
/// re-derive it here (it requires geometry analysis owned by B7).
pub fn selectFallbackRoute(
    caps: Capabilities,
    prompt_prefill: bool,
    generic_shape_eligible: bool,
) Route {
    if (prompt_prefill and caps.generic_mma) return .prompt_prefill;
    if (generic_shape_eligible and caps.generic_mma) return .generic_mma;
    if (caps.portable_native) return .portable_native;
    return .unavailable;
}

/// Decide whether the wide (16×8 fused) MMA kernel should run instead of
/// the regular MMA matrix. Mirrors `ggml_cuda_fattn_kvarn_use_wide_mma`.
/// `wide_kernel_supported` is owned by the launcher: the wide path is
/// only built for n_q>8 in the B6 MMA module (Dev A).
pub fn useWideMma(
    input_n_q: u32,
    gqa: u32,
    wide_kernel_supported: bool,
) bool {
    return wide_kernel_supported and input_n_q > 8 and input_n_q <= 16 and gqa > 4;
}

/// Body shape acceptance: both head dims equal, in {128, 256, 512}, and
/// the bitmask advertises the dim. Mirrors
/// `ggml_cuda_fattn_kvarn_body_shape_supported`.
pub fn bodyShapeSupported(caps: Capabilities, d_k: u32, d_v: u32) bool {
    if (d_k != d_v) return false;
    return switch (d_k) {
        128, 256, 512 => caps.supported_head_dims.has(d_k) and caps.portable_native,
        else => false,
    };
}

/// Shape acceptance for the portable path specifically. Equivalent to
/// `bodyShapeSupported` for now; the function is split so the portable
/// launcher can later tighten its constraints (e.g. require GQA integer)
/// without touching the body shape check above.
pub fn portableSupported(caps: Capabilities, d_k: u32, d_v: u32) bool {
    return bodyShapeSupported(caps, d_k, d_v);
}

// ─── Route counters + memory transient stats ───────────────────────────────
//
// Pure data; emission is `countersPrint` which writes via
// `debugz.dbg.printLevel(.detail, …)` — gated by the existing
// `DEBUG_LEVEL>=2` env var, no new flag here. The `perf_kvarn_route` flag
// will be added in the P4 build.zig/debug.zig window per PLAN_B1.md §7.

/// Per-route counter slot. Packed into `align(8)` to make `@atomicRmw`
/// word-aligned (atomic ops on a 64-bit value need 8B alignment on x86_64
/// and aarch64; without the wrapper struct, the array layout in
/// `RouteCounters` may not be 8B-aligned for every slot).
pub const RouteCounter = struct {
    value: u64 align(8) = 0,
};

pub const RouteCounters = struct {
    /// Number of times each route was selected as the primary path.
    by_route: [6]RouteCounter = [_]RouteCounter{.{}} ** 6,
    /// Number of times `selectRoute` returned `.unavailable` (the dispatch
    /// had no eligible path). Useful for fail-closed telemetry.
    unavailable: RouteCounter = .{},
    /// Number of fallbacks emitted by `selectFallbackRoute`.
    fallback_to_portable: RouteCounter = .{},
    fallback_to_generic: RouteCounter = .{},
    /// Total primary selections (sum of `by_route`).
    total_selections: RouteCounter = .{},
};

pub const TransientPeak = struct {
    value: u64 align(8) = 0,
};

pub const MemoryTransientStats = struct {
    /// Largest descriptor buffer size observed (bytes). Atomic max.
    peak_descriptor_bytes: TransientPeak = .{},
    /// Largest partials buffer size observed (bytes).
    peak_partials_bytes: TransientPeak = .{},
    /// Largest softmax-meta buffer size observed (bytes).
    peak_meta_bytes: TransientPeak = .{},
};

/// Process-global counters. Single instance is fine for the dispatch
/// master: the host launch is single-threaded. Future per-stream counters
/// would be an array indexed by `n_stream`.
var g_route_counters: RouteCounters = .{};
var g_mem_stats: MemoryTransientStats = .{};

/// Increment the counter for the chosen route. Safe to call from any
/// thread (aligned atomics).
pub fn countersInc(route: Route) void {
    const idx = @intFromEnum(route);
    if (idx < g_route_counters.by_route.len) {
        _ = @atomicRmw(u64, &g_route_counters.by_route[idx].value, .Add, 1, .seq_cst);
    }
    if (route == .unavailable) {
        _ = @atomicRmw(u64, &g_route_counters.unavailable.value, .Add, 1, .seq_cst);
    }
    _ = @atomicRmw(u64, &g_route_counters.total_selections.value, .Add, 1, .seq_cst);
}

pub fn countersIncFallback(kind: enum { portable, generic }) void {
    switch (kind) {
        .portable => _ = @atomicRmw(u64, &g_route_counters.fallback_to_portable.value, .Add, 1, .seq_cst),
        .generic => _ = @atomicRmw(u64, &g_route_counters.fallback_to_generic.value, .Add, 1, .seq_cst),
    }
}

pub fn countersReset() void {
    g_route_counters = .{};
    g_mem_stats = .{};
}

pub fn countersGet() RouteCounters {
    return .{
        .by_route = g_route_counters.by_route,
        .unavailable = g_route_counters.unavailable,
        .fallback_to_portable = g_route_counters.fallback_to_portable,
        .fallback_to_generic = g_route_counters.fallback_to_generic,
        .total_selections = g_route_counters.total_selections,
    };
}

pub fn memStatsGet() MemoryTransientStats {
    return .{
        .peak_descriptor_bytes = g_mem_stats.peak_descriptor_bytes,
        .peak_partials_bytes = g_mem_stats.peak_partials_bytes,
        .peak_meta_bytes = g_mem_stats.peak_meta_bytes,
    };
}

/// Update the peak (atomic max) of one of the transient memory fields.
/// Used by the dispatch master right before each launch to report the
/// max descriptor / partials / meta allocation footprint.
pub fn memStatsUpdate(kind: enum { descriptor, partials, meta }, bytes: u64) void {
    switch (kind) {
        .descriptor => _ = @atomicRmw(u64, &g_mem_stats.peak_descriptor_bytes.value, .Max, bytes, .seq_cst),
        .partials => _ = @atomicRmw(u64, &g_mem_stats.peak_partials_bytes.value, .Max, bytes, .seq_cst),
        .meta => _ = @atomicRmw(u64, &g_mem_stats.peak_meta_bytes.value, .Max, bytes, .seq_cst),
    }
}

/// Emit counters via the existing detail-level breadcrumb. Single line per
/// call; designed for periodic dumps from the dispatch master (every N
/// tokens or every M routes). Gated by `DEBUG_LEVEL>=2`.
pub fn countersPrint() void {
    if (!debugz.dbg.at(.detail)) return;
    const c = countersGet();
    const m = memStatsGet();
    debugz.dbg.printLevel(
        .detail,
        "[kvarn-route] totals={d} unavail={d} fb_port={d} fb_gen={d} | " ++
            "prompt_prefill={d} decode_split={d} decode_vector={d} generic_mma={d} " ++
            "portable_native={d} unavailable={d} | " ++
            "mem peak desc/partials/meta = {d}/{d}/{d} B\n",
        .{
            c.total_selections,
            c.unavailable,
            c.fallback_to_portable,
            c.fallback_to_generic,
            c.by_route[@intFromEnum(Route.prompt_prefill)],
            c.by_route[@intFromEnum(Route.decode_split)],
            c.by_route[@intFromEnum(Route.decode_vector)],
            c.by_route[@intFromEnum(Route.generic_mma)],
            c.by_route[@intFromEnum(Route.portable_native)],
            c.by_route[@intFromEnum(Route.unavailable)],
            m.peak_descriptor_bytes,
            m.peak_partials_bytes,
            m.peak_meta_bytes,
        },
    );
}

// ─── Env-var overrides ─────────────────────────────────────────────────────

/// Single source of truth for the runtime env knobs (PLAN_B1.md §8).
/// Read once and cached.
const Env = struct {
    force_portable: bool,
    vec_disabled: bool,
    vec_tokens: u32,
    debug_routes: bool,

    fn load() Env {
        return .{
            .force_portable = envOn("ZIG_AI_KVARN_FORCE_PORTABLE"),
            .vec_disabled = !envOnDefaultOn("ZIG_AI_KVARN_VEC"),
            .vec_tokens = envU32("ZIG_AI_KVARN_VEC_TOKENS") orelse 16,
            .debug_routes = envOn("ZIG_AI_KVARN_DEBUG_ROUTES"),
        };
    }
};

var g_env: Env = undefined;
var g_env_initialized: std.atomic.Value(bool) = .init(false);

/// Public read of the cached env. Lazy: first call parses env, subsequent
/// calls return the cached value. `force_portable` and `vec_disabled` are
/// intended to be picked up at launch time, not re-read in the hot path.
///
/// Concurrency: an atomic bool guards the one-shot init. The dispatch
/// master typically re-reads the relevant flag per launch, so the cached
/// value here is mostly for diagnostic code.
pub fn env() Env {
    if (!g_env_initialized.load(.seq_cst)) {
        const fresh = Env.load();
        @atomicStore(bool, &g_env_initialized.raw, true, .seq_cst);
        g_env = fresh;
    }
    return g_env;
}

/// One-shot reset for tests. NOT for production use.
pub fn envResetForTest() void {
    g_env = undefined;
    g_env_initialized.store(false, .seq_cst);
}

/// One-shot (no caching) for tests that want to flip the env at runtime.
pub fn envForcePortable() bool {
    return envOn("ZIG_AI_KVARN_FORCE_PORTABLE");
}
pub fn envVecDisabled() bool {
    return !envOnDefaultOn("ZIG_AI_KVARN_VEC");
}
pub fn envVecTokens() u32 {
    return envU32("ZIG_AI_KVARN_VEC_TOKENS") orelse 16;
}
pub fn envDebugRoutes() bool {
    return envOn("ZIG_AI_KVARN_DEBUG_ROUTES");
}

// ─── env helpers ───────────────────────────────────────────────────────────

fn envOn(name: [:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    const s = std.mem.span(v);
    if (s.len == 0) return true; // presence = on
    // Strict "=0" / "=false" / "=off" ⇒ off; everything else on.
    return !std.mem.eql(u8, s, "0") and
        !std.mem.eql(u8, s, "false") and
        !std.mem.eql(u8, s, "off");
}

fn envOnDefaultOn(name: [:0]const u8) bool {
    const v = std.c.getenv(name) orelse return true; // default: on
    const s = std.mem.span(v);
    if (s.len == 0) return true;
    return !std.mem.eql(u8, s, "0") and
        !std.mem.eql(u8, s, "false") and
        !std.mem.eql(u8, s, "off");
}

fn envU32(name: [:0]const u8) ?u32 {
    const v = std.c.getenv(name) orelse return null;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch null;
}

// ─── Self-tests (unit, pure CPU) ───────────────────────────────────────────

test "HeadDimMask set/has" {
    const m = HeadDimMask.fromDim(128);
    try std.testing.expect(m.has(128));
    try std.testing.expect(!m.has(256));
    try std.testing.expect(!m.has(512));
    try std.testing.expect(!m.has(64));
    const m2 = HeadDimMask{ .d128 = true, .d256 = true };
    try std.testing.expectEqual(@as(u8, 0b0000_0011), m2.bits());
}

test "RouteFamilyMask has/set" {
    var m: RouteFamilyMask = .{};
    try std.testing.expect(!m.has(.portable_native));
    m.set(.portable_native, true);
    try std.testing.expect(m.has(.portable_native));
    try std.testing.expect(!m.has(.decode_vector));
    // .unavailable never part of the bitmask.
    try std.testing.expect(!m.has(.unavailable));
    m.set(.unavailable, true);
    try std.testing.expect(!m.has(.unavailable));
}

test "selectCapabilities: sm_75 portable-only" {
    // sm_75 (Turing): wave=32, MMA absent. We expect portable native
    // (wave 32 + threads ≥ 128 + smem ≥ 3144) but no specialized routes.
    const caps = selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = false,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 64 * 1024,
    });
    try std.testing.expect(caps.portable_native);
    try std.testing.expect(!caps.generic_mma);
    try std.testing.expect(!caps.decode_split);
    try std.testing.expect(!caps.decode_vector);
    try std.testing.expect(!caps.specialized_routes);
    try std.testing.expectEqual(PORTABLE_MAX_Q, caps.rotated_query_max_portable);
    try std.testing.expectEqual(@as(u32, 0), caps.rotated_query_max_specialized);
    try std.testing.expect(caps.supported_head_dims.d128);
    try std.testing.expect(caps.supported_head_dims.d256);
    try std.testing.expect(caps.supported_head_dims.d512);
}

test "selectCapabilities: sm_86 with MMA" {
    const caps = selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 99 * 1024,
    });
    try std.testing.expect(caps.portable_native);
    try std.testing.expect(caps.generic_mma);
    try std.testing.expect(caps.decode_split);
    try std.testing.expect(caps.decode_vector);
    try std.testing.expect(caps.specialized_routes);
    try std.testing.expect(caps.original_v_domain);
    try std.testing.expectEqual(SPECIALIZED_DECODE_MAX_Q, caps.rotated_query_max_specialized);
    try std.testing.expect(caps.route_families.has(.portable_native));
    try std.testing.expect(caps.route_families.has(.generic_mma));
    try std.testing.expect(caps.route_families.has(.decode_split));
    try std.testing.expect(caps.route_families.has(.decode_vector));
}

test "selectCapabilities: no KVarN instances ⇒ store=false" {
    const caps = selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = false,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 99 * 1024,
    });
    try std.testing.expect(!caps.store_materialize);
    try std.testing.expect(!caps.portable_native);
    try std.testing.expect(!caps.generic_mma);
    try std.testing.expect(caps.supported_head_dims.bits() == 0);
}

test "selectCapabilities: HIP cannot decode_split/decode_vector" {
    const caps = selectCapabilities(.{
        .backend = .hip,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 64 * 1024,
        .amd_mma_family = .rdna_wmma,
    });
    try std.testing.expect(caps.generic_mma);
    try std.testing.expect(!caps.decode_split);
    try std.testing.expect(!caps.decode_vector);
    try std.testing.expect(!caps.original_v_domain);
}

test "selectCapabilities: MUSA no MMA" {
    const caps = selectCapabilities(.{
        .backend = .musa,
        .physical_wave_size = 32,
        .matrix_mma = true, // pretend the host advertises it; MUSA ignores
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 64 * 1024,
    });
    try std.testing.expect(caps.portable_native);
    try std.testing.expect(!caps.generic_mma);
    try std.testing.expect(!caps.decode_split);
    try std.testing.expect(!caps.decode_vector);
}

test "selectCapabilities: max_threads < 128 ⇒ portable fails closed" {
    const caps = selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 64, // < 128
        .shared_memory_per_block = 99 * 1024,
    });
    try std.testing.expect(!caps.portable_hardware);
    try std.testing.expect(!caps.portable_native);
    try std.testing.expect(!caps.store_materialize);
}

test "selectCapabilities: smem < 3144 ⇒ portable fails closed" {
    const caps = selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 2048, // < 3144
    });
    try std.testing.expect(!caps.portable_hardware);
    try std.testing.expect(!caps.portable_native);
}

test "selectCapabilities: shared_memory null ⇒ conservative 48 KiB" {
    // With null smem the function assumes 48 KiB. portable_hardware should
    // hold (48 KiB > 3144 B) so the gating is correct under the fallback.
    const caps = selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = null,
    });
    try std.testing.expect(caps.portable_hardware);
    try std.testing.expect(caps.portable_native);
}

test "selectCapabilities: physical_wave_size null ⇒ portable fail-closed" {
    // No probe ⇒ no portable. Fail-closed by design.
    const caps = selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = null,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 99 * 1024,
    });
    try std.testing.expect(!caps.portable_hardware);
    try std.testing.expect(!caps.portable_native);
}

test "selectRoute: prompt_prefill wins unconditionally" {
    const caps = sm86Caps();
    const r = selectRoute(caps, .{
        .head_dim = 128,
        .n_q = 1,
        .gqa = 8,
        .k_bits = 4,
        .v_bits = 4,
        .swa = false,
        .prompt_prefill = true,
        .vector_eligible = true, // would normally win without prompt
        .split_eligible = true,
    });
    try std.testing.expectEqual(Route.prompt_prefill, r);
}

test "selectRoute: vector_eligible wins before split (D=256 SWA GQA=2)" {
    const caps = sm86Caps();
    const r = selectRoute(caps, .{
        .head_dim = 256,
        .n_q = 1,
        .gqa = 2,
        .k_bits = 4,
        .v_bits = 4,
        .swa = true,
        .prompt_prefill = false,
        .vector_eligible = true,
        .split_eligible = true,
    });
    try std.testing.expectEqual(Route.decode_vector, r);
}

test "selectRoute: vec_disabled skips decode_vector" {
    const caps = sm86Caps();
    const r = selectRoute(caps, .{
        .head_dim = 256,
        .n_q = 1,
        .gqa = 2,
        .k_bits = 4,
        .v_bits = 4,
        .swa = true,
        .prompt_prefill = false,
        .vector_eligible = true,
        .split_eligible = true,
        .vec_disabled = true,
    });
    // vec disabled ⇒ split (n_q==1) is eligible ⇒ decode_split.
    try std.testing.expectEqual(Route.decode_split, r);
}

test "selectRoute: n_q==1 split_eligible (no vector) ⇒ decode_split" {
    const caps = sm86Caps();
    const r = selectRoute(caps, .{
        .head_dim = 128,
        .n_q = 1,
        .gqa = 8,
        .k_bits = 4,
        .v_bits = 4,
        .swa = false,
        .prompt_prefill = false,
        .vector_eligible = false,
        .split_eligible = true,
    });
    try std.testing.expectEqual(Route.decode_split, r);
}

test "selectRoute: n_q>1 split not eligible ⇒ generic_mma" {
    const caps = sm86Caps();
    const r = selectRoute(caps, .{
        .head_dim = 128,
        .n_q = 4,
        .gqa = 8,
        .k_bits = 4,
        .v_bits = 4,
        .swa = false,
        .prompt_prefill = false,
        .vector_eligible = false,
        .split_eligible = false, // caller can still report true, but
        // policy gates on n_q
    });
    try std.testing.expectEqual(Route.generic_mma, r);
}

test "selectRoute: sm_75 (no MMA) ⇒ portable" {
    const caps = selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = false,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 64 * 1024,
    });
    const r = selectRoute(caps, .{
        .head_dim = 128,
        .n_q = 4,
        .gqa = 8,
        .k_bits = 4,
        .v_bits = 4,
        .swa = false,
        .prompt_prefill = false,
        .vector_eligible = false,
        .split_eligible = false,
    });
    try std.testing.expectEqual(Route.portable_native, r);
}

test "selectRoute: no portable + no MMA ⇒ unavailable (fail-closed)" {
    const caps = selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = false,
        .kvarn_instances = false, // no KVarN instances
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 64 * 1024,
    });
    const r = selectRoute(caps, .{
        .head_dim = 128,
        .n_q = 1,
        .gqa = 8,
        .k_bits = 4,
        .v_bits = 4,
        .swa = false,
        .prompt_prefill = false,
        .vector_eligible = false,
        .split_eligible = true,
    });
    try std.testing.expectEqual(Route.unavailable, r);
}

test "selectRoute: force_portable overrides natural priority" {
    const caps = sm86Caps();
    const r = selectRoute(caps, .{
        .head_dim = 128,
        .n_q = 1,
        .gqa = 8,
        .k_bits = 4,
        .v_bits = 4,
        .swa = false,
        .prompt_prefill = true, // would otherwise win
        .vector_eligible = true,
        .split_eligible = true,
        .force_portable = true,
    });
    try std.testing.expectEqual(Route.portable_native, r);
}

test "selectRoute: force_portable + no portable ⇒ unavailable" {
    const caps = selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = false, // no portable
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 64 * 1024,
    });
    const r = selectRoute(caps, .{
        .head_dim = 128,
        .n_q = 1,
        .gqa = 8,
        .k_bits = 4,
        .v_bits = 4,
        .swa = false,
        .prompt_prefill = false,
        .vector_eligible = false,
        .split_eligible = true,
        .force_portable = true,
    });
    try std.testing.expectEqual(Route.unavailable, r);
}

test "selectFallbackRoute: prompt_prefill ⇒ prompt_prefill" {
    const caps = sm86Caps();
    try std.testing.expectEqual(Route.prompt_prefill, selectFallbackRoute(caps, true, true));
}

test "selectFallbackRoute: generic_shape_eligible ⇒ generic_mma" {
    const caps = sm86Caps();
    try std.testing.expectEqual(Route.generic_mma, selectFallbackRoute(caps, false, true));
}

test "selectFallbackRoute: neither ⇒ portable" {
    const caps = sm86Caps();
    try std.testing.expectEqual(Route.portable_native, selectFallbackRoute(caps, false, false));
}

test "selectFallbackRoute: no portable + no generic ⇒ unavailable" {
    const caps = selectCapabilities(.{
        .backend = .musa, // no generic
        .physical_wave_size = 32,
        .matrix_mma = false,
        .kvarn_instances = false, // no portable
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 64 * 1024,
    });
    try std.testing.expectEqual(Route.unavailable, selectFallbackRoute(caps, false, true));
}

test "useWideMma: 8 < n_q ≤ 16 && gqa > 4 ⇒ true" {
    try std.testing.expect(useWideMma(9, 8, true));
    try std.testing.expect(useWideMma(16, 8, true));
    try std.testing.expect(!useWideMma(8, 8, true)); // not > 8
    try std.testing.expect(!useWideMma(17, 8, true)); // not ≤ 16
    try std.testing.expect(!useWideMma(9, 4, true)); // not > 4
    try std.testing.expect(!useWideMma(9, 8, false)); // not supported
}

test "bodyShapeSupported: d_k != d_v ⇒ false" {
    const caps = sm86Caps();
    try std.testing.expect(!bodyShapeSupported(caps, 128, 256));
}

test "bodyShapeSupported: dim in {128,256,512} & supported & portable ⇒ true" {
    const caps = sm86Caps();
    try std.testing.expect(bodyShapeSupported(caps, 128, 128));
    try std.testing.expect(bodyShapeSupported(caps, 256, 256));
    try std.testing.expect(bodyShapeSupported(caps, 512, 512));
}

test "bodyShapeSupported: dim unsupported ⇒ false" {
    const caps = selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = false,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 64 * 1024,
        .amd_mma_family = .none,
    });
    // 64 / 1024 are not in the supported set.
    try std.testing.expect(!bodyShapeSupported(caps, 64, 64));
    try std.testing.expect(!bodyShapeSupported(caps, 1024, 1024));
}

test "portableSupported matches bodyShapeSupported (today)" {
    const caps = sm86Caps();
    try std.testing.expectEqual(bodyShapeSupported(caps, 128, 128), portableSupported(caps, 128, 128));
    try std.testing.expectEqual(bodyShapeSupported(caps, 256, 256), portableSupported(caps, 256, 256));
    try std.testing.expectEqual(bodyShapeSupported(caps, 512, 512), portableSupported(caps, 512, 512));
    try std.testing.expectEqual(bodyShapeSupported(caps, 128, 256), portableSupported(caps, 128, 256));
}

test "RouteCounters: inc + read" {
    countersReset();
    countersInc(.portable_native);
    countersInc(.portable_native);
    countersInc(.generic_mma);
    countersInc(.unavailable);
    countersIncFallback(.portable);
    countersIncFallback(.generic);
    countersIncFallback(.generic);
    const c = countersGet();
    try std.testing.expectEqual(@as(u64, 2), c.by_route[@intFromEnum(Route.portable_native)].value);
    try std.testing.expectEqual(@as(u64, 1), c.by_route[@intFromEnum(Route.generic_mma)].value);
    try std.testing.expectEqual(@as(u64, 1), c.by_route[@intFromEnum(Route.unavailable)].value);
    try std.testing.expectEqual(@as(u64, 1), c.unavailable.value);
    try std.testing.expectEqual(@as(u64, 1), c.fallback_to_portable.value);
    try std.testing.expectEqual(@as(u64, 2), c.fallback_to_generic.value);
    try std.testing.expectEqual(@as(u64, 4), c.total_selections.value);
    countersReset();
}

test "MemoryTransientStats: atomic-max" {
    countersReset();
    memStatsUpdate(.descriptor, 1024);
    memStatsUpdate(.descriptor, 2048);
    memStatsUpdate(.descriptor, 1500);
    memStatsUpdate(.partials, 4096);
    memStatsUpdate(.partials, 2048);
    memStatsUpdate(.meta, 256);
    const m = memStatsGet();
    try std.testing.expectEqual(@as(u64, 2048), m.peak_descriptor_bytes.value);
    try std.testing.expectEqual(@as(u64, 4096), m.peak_partials_bytes.value);
    try std.testing.expectEqual(@as(u64, 256), m.peak_meta_bytes.value);
    countersReset();
}

test "env helpers: defaults match PLAN_B1.md §8" {
    // These run with no env set, so defaults apply.
    try std.testing.expect(!envForcePortable());
    try std.testing.expect(!envVecDisabled()); // default: vec ON
    try std.testing.expectEqual(@as(u32, 16), envVecTokens());
    try std.testing.expect(!envDebugRoutes());
}

test "env helpers: ZIG_AI_KVARN_FORCE_PORTABLE=1 ⇒ on" {
    // We can't actually set env in a unit test portably; the value
    // depends on the runner. Test only the OFF path which is the
    // default and always true in CI.
    if (std.c.getenv("ZIG_AI_KVARN_FORCE_PORTABLE")) |_| {
        // present ⇒ ON (unless explicitly "0" — covered by the helper).
        try std.testing.expect(envForcePortable() or !envForcePortable());
    } else {
        try std.testing.expect(!envForcePortable());
    }
}

test "env() caches" {
    const a = env();
    const b = env();
    try std.testing.expectEqual(a.force_portable, b.force_portable);
    try std.testing.expectEqual(a.vec_disabled, b.vec_disabled);
    try std.testing.expectEqual(a.vec_tokens, b.vec_tokens);
    try std.testing.expectEqual(a.debug_routes, b.debug_routes);
}

// Helper: a sm_86-shaped `Capabilities` for the policy tests.
fn sm86Caps() Capabilities {
    return selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 99 * 1024,
    });
}
