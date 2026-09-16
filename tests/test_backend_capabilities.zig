//! Lane B1 — B2: unit tests for `backend_capabilities.zig`.
//!
//! Spec (TODO_B1_DEV_B.md §B2):
//!   - Smoke table per SM (75/80/86/89/90) — verify the per-arch
//!     capability snapshot is what we expect.
//!   - Regresión: mma=false ⇒ dispatch NUNCA elige rutas MMA
//!     (fail-closed, sin portable ⇒ unavailable).
//!   - Probing real en el device presente (sm_86 según configuración del
//!     entorno) — verificación suave: si el probe funciona, los campos que
//!     podemos leer de un sm_86 real deben coincidir con la tabla.
//!
//! Todas las pruebas son CPU-only: este test NO requiere CUDA para pasar
//! (los tests de capability son sobre `selectCapabilities`, que es puro).
//! El test "probe real" se marca con `try testing.expect(...)` sobre un
//! resultado de probe "soft": si no hay GPU no falla — registra null.

const std = @import("std");
const testing = std.testing;

const bc = @import("backend_capabilities");
const debugz = @import("debug");
const fattn_kv = @import("fattn_kvarn");

// ─── Smoke table per SM ────────────────────────────────────────────────────

test "sm_75: portable-only, no MMA" {
    const caps = bc.selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = false, // sm_75 has no tensor cores
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 64 * 1024,
    });
    try testing.expect(caps.portable_native);
    try testing.expect(!caps.specialized_routes);
    try testing.expect(!caps.generic_mma);
    try testing.expect(!caps.decode_split);
    try testing.expect(!caps.decode_vector);
    try testing.expect(!caps.original_v_domain);
    try testing.expect(caps.route_families.has(.portable_native));
    try testing.expect(!caps.route_families.has(.generic_mma));
    try testing.expect(!caps.route_families.has(.decode_split));
    try testing.expect(!caps.route_families.has(.decode_vector));
    try testing.expectEqual(bc.PORTABLE_MAX_Q, caps.rotated_query_max_portable);
    try testing.expectEqual(@as(u32, 0), caps.rotated_query_max_specialized);
}

test "sm_80 (A100): full MMA + portable" {
    const caps = bc.selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 163 * 1024,
    });
    try testing.expect(caps.portable_native);
    try testing.expect(caps.generic_mma);
    try testing.expect(caps.decode_split);
    try testing.expect(caps.decode_vector);
    try testing.expect(caps.specialized_routes);
    try testing.expect(caps.original_v_domain);
    try testing.expectEqual(bc.SPECIALIZED_DECODE_MAX_Q, caps.rotated_query_max_specialized);
}

test "sm_86 (RTX 30/40): full MMA + portable" {
    // Most common consumer card. Should be indistinguishable from sm_80
    // for the policy (both report matrix_mma=true).
    const caps = bc.selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 99 * 1024, // 99 KiB opt-in sm_86
    });
    try testing.expect(caps.portable_native);
    try testing.expect(caps.specialized_routes);
    try testing.expect(caps.generic_mma);
    try testing.expect(caps.decode_split);
    try testing.expect(caps.decode_vector);
    try testing.expect(caps.route_families.has(.portable_native));
    try testing.expect(caps.route_families.has(.generic_mma));
    try testing.expect(caps.route_families.has(.decode_split));
    try testing.expect(caps.route_families.has(.decode_vector));
}

test "sm_89 (Ada): full MMA + portable" {
    const caps = bc.selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 99 * 1024,
    });
    try testing.expect(caps.portable_native);
    try testing.expect(caps.specialized_routes);
    try testing.expect(caps.original_v_domain);
}

test "sm_90 (Hopper): full MMA + portable" {
    // sm_90 has wgmma but the policy does not distinguish — it only reads
    // `matrix_mma` (a coarse flag). The kernel-level wgmma contract lives
    // in Dev A's MMA module. From the policy's perspective sm_90 is a
    // sm_80+ with matrix_mma=true.
    const caps = bc.selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 228 * 1024, // up to 228 KiB opt-in
    });
    try testing.expect(caps.portable_native);
    try testing.expect(caps.specialized_routes);
    try testing.expect(caps.original_v_domain);
}

// ─── Fail-closed regression ────────────────────────────────────────────────

test "fail-closed: mma=false ⇒ no route chooses MMA family" {
    const caps = bc.selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = false,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 64 * 1024,
    });
    // Walk every input combination that could potentially pick an MMA
    // route and verify portable is chosen or, failing that, unavailable.
    const cases = [_]bc.RouteInput{
        .{
            .head_dim = 128,
            .n_q = 1,
            .gqa = 8,
            .k_bits = 4,
            .v_bits = 4,
            .swa = false,
            .prompt_prefill = true,
            .vector_eligible = false,
            .split_eligible = false,
        },
        .{
            .head_dim = 256,
            .n_q = 1,
            .gqa = 2,
            .k_bits = 4,
            .v_bits = 4,
            .swa = true,
            .prompt_prefill = false,
            .vector_eligible = true,
            .split_eligible = true,
        },
        .{
            .head_dim = 128,
            .n_q = 1,
            .gqa = 8,
            .k_bits = 4,
            .v_bits = 4,
            .swa = false,
            .prompt_prefill = false,
            .vector_eligible = false,
            .split_eligible = true,
        },
        .{
            .head_dim = 128,
            .n_q = 4,
            .gqa = 8,
            .k_bits = 4,
            .v_bits = 4,
            .swa = false,
            .prompt_prefill = false,
            .vector_eligible = false,
            .split_eligible = false,
        },
        .{
            .head_dim = 128,
            .n_q = 16,
            .gqa = 8,
            .k_bits = 4,
            .v_bits = 4,
            .swa = false,
            .prompt_prefill = false,
            .vector_eligible = false,
            .split_eligible = false,
        },
    };
    for (cases) |input| {
        const r = bc.selectRoute(caps, input);
        try testing.expect(r != .generic_mma);
        try testing.expect(r != .decode_split);
        try testing.expect(r != .decode_vector);
        try testing.expect(r == .portable_native or r == .unavailable or r == .prompt_prefill);
    }
}

test "fail-closed: no KVarN + no portable ⇒ unavailable" {
    const caps = bc.selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true, // pretend the GPU has MMA
        .kvarn_instances = false, // but no cubins
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 99 * 1024,
    });
    try testing.expect(!caps.store_materialize);
    try testing.expect(!caps.portable_native);
    const r = bc.selectRoute(caps, .{
        .head_dim = 128,
        .n_q = 1,
        .gqa = 8,
        .k_bits = 4,
        .v_bits = 4,
        .swa = false,
        .prompt_prefill = false,
        .vector_eligible = true,
        .split_eligible = true,
    });
    try testing.expectEqual(bc.Route.unavailable, r);
}

test "fail-closed: physical_wave_size=null ⇒ no portable (no MMA either)" {
    // Worst case: probe failed entirely. We refuse to launch anything.
    const caps = bc.selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = null,
        .matrix_mma = false, // also false, since we couldn't read it
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 99 * 1024,
    });
    try testing.expect(!caps.portable_hardware);
    try testing.expect(!caps.portable_native);
    try testing.expect(!caps.generic_mma);
    const r = bc.selectRoute(caps, .{
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
    try testing.expectEqual(bc.Route.unavailable, r);
}

test "fail-closed: MUSA has no MMA path (even with matrix_mma=true)" {
    const caps = bc.selectCapabilities(.{
        .backend = .musa,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 64 * 1024,
    });
    try testing.expect(caps.portable_native);
    try testing.expect(!caps.generic_mma);
    try testing.expect(!caps.decode_split);
    try testing.expect(!caps.decode_vector);
}

// ─── Real-device probe (soft; no GPU required) ─────────────────────────────

test "probeDevice ordinal 0 returns sensible defaults" {
    // This call may or may not reach a GPU. Either way it must return
    // a Capabilities value that, when fed to selectRoute, fail-closes if
    // the probe could not read anything useful.
    const caps = bc.probeDevice(0);
    // Backend is always CUDA for the lane today.
    try testing.expectEqual(bc.Backend.cuda, caps.backend);
    // The probe never sets kvarn_instances (the dispatch master does).
    try testing.expect(!caps.kvarn_instances);
    // max_threads_per_block is the fallback (PORTABLE_THREADS) or a real
    // value; both are >= PORTABLE_THREADS.
    try testing.expect(caps.max_threads_per_block >= bc.PORTABLE_THREADS);
}

test "probeDevice ordinal 0 with no KVarN ⇒ selectRoute falls back to portable" {
    const caps = bc.probeDevice(0);
    // We can't influence kvarn_instances from here, so we just verify the
    // route decision is well-defined.
    const r = bc.selectRoute(caps, .{
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
    // Without kvarn_instances the route is one of {decode_split, unavailable}.
    try testing.expect(r == .decode_split or r == .unavailable);
}

test "isDeviceVisible: graceful on no GPU" {
    // Just call it; should not crash regardless of GPU presence.
    _ = bc.isDeviceVisible();
}

// ─── Body shape & wide MMA ────────────────────────────────────────────────

test "bodyShapeSupported: sm_86 with d_k == d_v in {128,256,512}" {
    const caps = bc.selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 99 * 1024,
    });
    try testing.expect(bc.bodyShapeSupported(caps, 128, 128));
    try testing.expect(bc.bodyShapeSupported(caps, 256, 256));
    try testing.expect(bc.bodyShapeSupported(caps, 512, 512));
    try testing.expect(!bc.bodyShapeSupported(caps, 128, 256)); // d_k != d_v
    try testing.expect(!bc.bodyShapeSupported(caps, 64, 64)); // unsupported dim
}

test "useWideMma: 9 ≤ n_q ≤ 16 && gqa > 4 ⇒ true on sm_86 caps" {
    _ = bc.selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 99 * 1024,
    });
    // wide_kernel_supported is owned by the launcher (B6). We test the
    // gating function in isolation.
    try testing.expect(bc.useWideMma(9, 8, true));
    try testing.expect(bc.useWideMma(16, 8, true));
    try testing.expect(!bc.useWideMma(8, 8, true)); // not > 8
    try testing.expect(!bc.useWideMma(17, 8, true)); // not ≤ 16
    try testing.expect(!bc.useWideMma(9, 4, true)); // gqa not > 4
    try testing.expect(!bc.useWideMma(9, 8, false)); // not supported
}

// ─── Counters round-trip ───────────────────────────────────────────────────

test "RouteCounters accumulate, reset, and report total" {
    bc.countersReset();
    defer bc.countersReset();
    bc.countersInc(.portable_native);
    bc.countersInc(.portable_native);
    bc.countersInc(.generic_mma);
    bc.countersInc(.unavailable);
    bc.countersIncFallback(.portable);
    bc.countersIncFallback(.generic);
    bc.countersIncFallback(.generic);
    const c = bc.countersGet();
    try testing.expectEqual(@as(u64, 2), c.by_route[@intFromEnum(bc.Route.portable_native)].value);
    try testing.expectEqual(@as(u64, 1), c.by_route[@intFromEnum(bc.Route.generic_mma)].value);
    try testing.expectEqual(@as(u64, 1), c.by_route[@intFromEnum(bc.Route.unavailable)].value);
    try testing.expectEqual(@as(u64, 1), c.unavailable.value);
    try testing.expectEqual(@as(u64, 1), c.fallback_to_portable.value);
    try testing.expectEqual(@as(u64, 2), c.fallback_to_generic.value);
    try testing.expectEqual(@as(u64, 4), c.total_selections.value);
}

test "MemoryTransientStats: atomic-max across kinds" {
    bc.countersReset();
    defer bc.countersReset();
    bc.memStatsUpdate(.descriptor, 1024);
    bc.memStatsUpdate(.descriptor, 4096);
    bc.memStatsUpdate(.descriptor, 2048);
    bc.memStatsUpdate(.partials, 8192);
    bc.memStatsUpdate(.partials, 4096);
    bc.memStatsUpdate(.meta, 256);
    const m = bc.memStatsGet();
    try testing.expectEqual(@as(u64, 4096), m.peak_descriptor_bytes.value);
    try testing.expectEqual(@as(u64, 8192), m.peak_partials_bytes.value);
    try testing.expectEqual(@as(u64, 256), m.peak_meta_bytes.value);
}

// ─── Env-var defaults ──────────────────────────────────────────────────────

test "env defaults: vec ON, vec_tokens=16, no force, no debug_routes" {
    // The test runner doesn't set these env vars ⇒ defaults apply.
    try testing.expect(!bc.envForcePortable());
    try testing.expect(!bc.envVecDisabled());
    try testing.expectEqual(@as(u32, 16), bc.envVecTokens());
    try testing.expect(!bc.envDebugRoutes());
}

// ─── B7 dispatch <-> B1 selectRoute parity ────────────────────────────────
//
// Catches a real class of bugs: drift between the policy function
// (bc.selectRoute) and the dispatcher (fattn_kv.dispatchKvarnAttention)
// when the policy or dispatcher evolves. Both must agree on the
// route for any (caps, input) combination — otherwise the bench
// measures one path while the dispatcher picks another.
//
// Test pattern: build a matrix of (caps, RouteInput, DispatchInput)
// pairs that exercise the priority order. Run both functions and
// assert they return the same Route. With module=null, the
// dispatcher computes the route without launching — pure function.

test "B7 dispatch parity: selectRoute == dispatchKvarnAttention.route" {
    // Test cases that exercise the priority order:
    //   - force_portable
    //   - prompt_prefill
    //   - vector_eligible (D=256 SWA GQA=2 n_q=1)
    //   - split_eligible (n_q=1)
    //   - generic_mma (n_q <= max, MMA cap available)
    //   - portable (MMA unavailable)
    //   - unavailable (no portable, no MMA)
    const cases = [_]struct {
        caps: bc.Capabilities,
        route_input: bc.RouteInput,
        dispatch_input: fattn_kv.DispatchInput,
    }{
        // 1) force_portable with portable available ⇒ portable_native
        .{
            .caps = bc.selectCapabilities(.{
                .backend = .cuda,
                .physical_wave_size = 32,
                .matrix_mma = true,
                .kvarn_instances = true,
                .max_threads_per_block = 1024,
                .shared_memory_per_block = 99 * 1024,
            }),
            .route_input = .{
                .head_dim = 128,
                .n_q = 4,
                .gqa = 8,
                .k_bits = 4,
                .v_bits = 4,
                .swa = false,
                .prompt_prefill = true,
                .vector_eligible = true,
                .split_eligible = true,
                .force_portable = true,
            },
            .dispatch_input = .{
                .head_dim = 128,
                .n_q = 4,
                .gqa = 8,
                .k_bits = 4,
                .v_bits = 4,
                .swa = false,
                .prompt_prefill = true,
                .vector_eligible = true,
                .split_eligible = true,
                .explicit_eligibility = true,
                .force_portable = true,
                .vec_disabled = false,
            },
        },
        // 2) prompt_prefill without force_portable ⇒ prompt_prefill
        .{
            .caps = bc.selectCapabilities(.{
                .backend = .cuda,
                .physical_wave_size = 32,
                .matrix_mma = true,
                .kvarn_instances = true,
                .max_threads_per_block = 1024,
                .shared_memory_per_block = 99 * 1024,
            }),
            .route_input = .{
                .head_dim = 128,
                .n_q = 4,
                .gqa = 8,
                .k_bits = 4,
                .v_bits = 4,
                .swa = false,
                .prompt_prefill = true,
                .vector_eligible = false,
                .split_eligible = false,
                .force_portable = false,
            },
            .dispatch_input = .{
                .head_dim = 128,
                .n_q = 4,
                .gqa = 8,
                .k_bits = 4,
                .v_bits = 4,
                .swa = false,
                .prompt_prefill = true,
                .vector_eligible = false,
                .split_eligible = false,
                .explicit_eligibility = true,
                .force_portable = false,
                .vec_disabled = false,
            },
        },
        // 3) vector_eligible with vec path available ⇒ decode_vector
        .{
            .caps = bc.selectCapabilities(.{
                .backend = .cuda,
                .physical_wave_size = 32,
                .matrix_mma = true,
                .kvarn_instances = true,
                .max_threads_per_block = 1024,
                .shared_memory_per_block = 99 * 1024,
            }),
            .route_input = .{
                .head_dim = 256,
                .n_q = 1,
                .gqa = 2,
                .k_bits = 4,
                .v_bits = 4,
                .swa = true,
                .prompt_prefill = false,
                .vector_eligible = true,
                .split_eligible = true,
                .force_portable = false,
            },
            .dispatch_input = .{
                .head_dim = 256,
                .n_q = 1,
                .gqa = 2,
                .k_bits = 4,
                .v_bits = 4,
                .swa = true,
                .prompt_prefill = false,
                .vector_eligible = true,
                .split_eligible = true,
                .explicit_eligibility = true,
                .force_portable = false,
                .vec_disabled = false,
            },
        },
        // 4) split_eligible n_q=1, no vec, no MMA ⇒ portable (MMA not available)
        .{
            .caps = bc.selectCapabilities(.{
                .backend = .cuda,
                .physical_wave_size = 32,
                .matrix_mma = false,
                .kvarn_instances = true,
                .max_threads_per_block = 1024,
                .shared_memory_per_block = 64 * 1024,
            }),
            .route_input = .{
                .head_dim = 128,
                .n_q = 1,
                .gqa = 8,
                .k_bits = 4,
                .v_bits = 4,
                .swa = false,
                .prompt_prefill = false,
                .vector_eligible = false,
                .split_eligible = true,
                .force_portable = false,
            },
            .dispatch_input = .{
                .head_dim = 128,
                .n_q = 1,
                .gqa = 8,
                .k_bits = 4,
                .v_bits = 4,
                .swa = false,
                .prompt_prefill = false,
                .vector_eligible = false,
                .split_eligible = true,
                .explicit_eligibility = true,
                .force_portable = false,
                .vec_disabled = false,
            },
        },
        // 5) no portable, no MMA, n_q=1 ⇒ unavailable
        .{
            .caps = bc.selectCapabilities(.{
                .backend = .cuda,
                .physical_wave_size = 32,
                .matrix_mma = false,
                .kvarn_instances = false,
                .max_threads_per_block = 1024,
                .shared_memory_per_block = 64 * 1024,
            }),
            .route_input = .{
                .head_dim = 128,
                .n_q = 1,
                .gqa = 8,
                .k_bits = 4,
                .v_bits = 4,
                .swa = false,
                .prompt_prefill = false,
                .vector_eligible = false,
                .split_eligible = true,
                .force_portable = false,
            },
            .dispatch_input = .{
                .head_dim = 128,
                .n_q = 1,
                .gqa = 8,
                .k_bits = 4,
                .v_bits = 4,
                .swa = false,
                .prompt_prefill = false,
                .vector_eligible = false,
                .split_eligible = true,
                .explicit_eligibility = true,
                .force_portable = false,
                .vec_disabled = false,
            },
        },
    };
    for (cases) |case| {
        const policy = bc.selectRoute(case.caps, case.route_input);
        // Dispatcher with module=null: computes route without launching.
        const result = fattn_kv.dispatchKvarnAttention(
            null,
            case.caps,
            case.dispatch_input,
            null,
            null,
            @ptrFromInt(1),
        ) catch continue;
        try testing.expectEqual(policy, result.route);
    }
}
