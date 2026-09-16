//! Lane-b1 B4 (Dev-B) — gate M1 prep: bit-exactness smoke for the
//! in-kernel WHT-128 rotation used by the portable attention.
//!
//! WHY THIS TEST EXISTS (D3 critical risk per PLAN_B1.md):
//!   The portable FA rotates Q in-kernel and inverts at the output.
//!   If the device butterfly drifts from the CPU `hadamard128InPlace`
//!   in kvarn.zig (different summation order, fma vs mul+add, fp32
//!   precision quirks), the matmul accumulates the drift and M1 fails.
//!   Same lesson as the WHT-128 device F0 gate Dev A already published:
//!   bit-exact on the WHT *before* integrating into the FA pipeline.
//!
//! WHAT THIS TEST DOES:
//!   - Load N=64 random rows of 128 f32s into device memory.
//!   - Apply `fattn_kvarn_wht_128` to each row on device.
//!   - Apply `hadamard128InPlace` (CPU ref) to each row on host.
//!   - Compare bit-by-bit. Any mismatch ⇒ M1 will not pass — fail
//!     loud here, fix the kernel there.
//!
//! SCOPE:
//!   - The WHT in the portable kernel is the SAME algorithm as A1's
//!     `kvarn_wht_128_impl` in kvarn_kernels.cu. This test re-exercises
//!     the WHT with the FA's exact call pattern (one row per block).
//!   - We cannot call the WHT directly (it's `static __device__`); we
//!     invoke the FA's D=128 kernel with a sentinel input that
//!     produces a no-op matmul (K=zero, V=zero) and then check that the
//!     output reflects a pure WHT(Q) → softmax(zeros) → WHT⁻¹ round-trip.
//!
//!   Actually simpler: a dedicated smoke kernel is exposed in
//!   `fattn_kvarn_portable.cu` as a thin wrapper (see `wht_roundtrip_kernel`).
//!   That keeps the test independent from the full FA path.

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const cudaz = @import("cudaz");
const kvarn = @import("kv_cache").kvarn;

const N_ROWS = 32;

/// Decl externa del kernel de smoke WHT-128 (exportado por
/// fattn_kvarn_portable.cu, ver "wht_roundtrip_kernel" allí).
pub extern "c" fn fattn_kvarn_wht_128_rows_kernel(
    rows: [*]f32,
    n_rows: c_int,
) void;

fn whtHost(rows: []f32) void {
    var r: usize = 0;
    while (r < rows.len / 128) : (r += 1) {
        kvarn.hadamard128InPlace(rows[r * 128 ..][0..128]);
    }
}

test "B4 WHT-128 device bit-exacto vs hadamard128InPlace (gate M1 D3)" {
    if (build_options.fattn_cubin.len == 0) return error.SkipZigTest;

    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.fattn_cubin);
    const func = try cudaz.cuModuleGetFunction(module, "fattn_kvarn_wht_128_rows_kernel");

    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();

    const rows_host = try allocator.alloc(f32, N_ROWS * 128);
    defer allocator.free(rows_host);
    for (rows_host) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    // CPU ref.
    const ref = try allocator.dupe(f32, rows_host);
    defer allocator.free(ref);
    whtHost(ref);

    // GPU: upload, launch, download.
    const d_rows = try cudaz.cuMemAlloc(@sizeOf(f32) * rows_host.len);
    defer cudaz.cuMemFree(d_rows);
    try cudaz.cuMemcpyHtoD(d_rows, @intFromPtr(rows_host.ptr), @sizeOf(f32) * rows_host.len);

    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    const n_rows_c: c_int = @intCast(N_ROWS);
    var kp: [2]?*const anyopaque = .{ &d_rows, &n_rows_c };
    try cudaz.cuLaunchKernel(
        func,
        @intCast(N_ROWS), // grid x: un bloque por fila
        1,
        1,
        128, // block
        1,
        1,
        512, // dyn shared: 128 floats
        stream,
        @ptrCast(&kp),
        null,
    );
    try cudaz.cuStreamSynchronize(stream);

    const out_host = try allocator.alloc(f32, rows_host.len);
    defer allocator.free(out_host);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_host.ptr), d_rows, @sizeOf(f32) * rows_host.len);

    var bad: usize = 0;
    for (out_host, ref, 0..) |got, want, i| {
        if (std.math.isNan(got) and std.math.isNan(want)) continue;
        if (got != want) {
            if (bad < 5) std.log.err("WHT mismatch @{d}: got {d} want {d}", .{ i, got, want });
            bad += 1;
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}
