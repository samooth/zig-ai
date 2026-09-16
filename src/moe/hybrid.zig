//! Hybrid execution configuration for FreeToken's q* policy.
//!
//! Implements bandwidth-adaptive hybrid execution splitting expert layers
//! between GPU (via PCIe) and CPU based on bandwidth ratios:
//!   q* = pcie_bw / (pcie_bw + cpu_bw)
//! where PCIe time for F experts = F * B / pcie_bw
//! and CPU time for M-F experts = (M - F) * B / cpu_bw
const std = @import("std");
const debugz = @import("debug");
const build_options = @import("build_options");
const cuda_runtime = @import("cuda_runtime");

extern "c" fn hybridSplitKernelLauncher(num_missing: u32, fetch_frac_q16: u32, result: *u32) void;

/// Configuration for hybrid CPU-GPU expert execution.
pub const HybridConfig = struct {
    /// Fetch fraction as Q16 fixed-point number (q* = pcie_bw/(pcie_bw+cpu_bw) * 65536)
    /// Represents fraction of experts to fetch via PCIe (GPU), remainder computed on CPU
    fetch_fraction_q16: u32,
    /// CPU memory bandwidth in GB/s
    cpu_bw_gbps: f32,
    /// PCIe memory bandwidth in GB/s
    pcie_bw_gbps: f32,
};

/// Computes the fetch fraction Q16 value from CPU and PCIe bandwidths.
/// Implements FreeToken's q* policy: q* = pcie_bw/(pcie_bw+cpu_bw)
pub fn computeFetchFraction(cpu_bw_gbps: f32, pcie_bw_gbps: f32) u32 {
    const total_bw = cpu_bw_gbps + pcie_bw_gbps;
    // Avoid division by zero - if both are zero, default to 0 (no fetch)
    if (total_bw == 0.0) return 0;
    const frac = pcie_bw_gbps / total_bw;
    // Convert to Q16 fixed-point and round
    return @intFromFloat(@round(frac * 65536.0));
}

/// Performs hybrid split calculation for expert distribution.
/// Given num_missing experts to compute and fetch_fraction_q16 (Q16 fixed-point),
/// returns the optimal number of experts to fetch via PCIe (rest computed on CPU).
/// Uses minimax approach to balance PCIe and CPU time.
pub fn hybridSplitKernel(num_missing: u32, fetch_frac_q16: u32) u32 {
    // Handle edge cases
    if (num_missing == 0) return 0;
    if (fetch_frac_q16 == 0) return 0;  // Fetch none
    if (fetch_frac_q16 == 65536) return num_missing;  // Fetch all

    // Calculate number of experts to fetch via PCIe
    const lo = (num_missing * fetch_frac_q16) >> 16;
    
    // Minimax: choose integer that minimizes max(PCIe_time, CPU_time)
    // PCIe time proportional to fetch_frac_q16 * num_fetched
    // CPU time proportional to (0x10000 - fetch_frac_q16) * num_computed
    const tmp_lo = lo * (0x10000 - @as(u32, fetch_frac_q16));
    const tmp_hi = (num_missing - lo) * @as(u32, fetch_frac_q16);
    const cost_lo = if (tmp_lo > tmp_hi) tmp_lo else tmp_hi;
    const tmp_lo2 = (lo + 1) * (0x10000 - @as(u32, fetch_frac_q16));
    const tmp_hi2 = (num_missing - lo - 1) * @as(u32, fetch_frac_q16);
    const cost_hi = if (tmp_lo2 > tmp_hi2) tmp_lo2 else tmp_hi2;
    
    return if (cost_lo <= cost_hi) lo else lo + 1;
}

/// Launches the CUDA kernel to compute the hybrid split.
/// This function mirrors the logic of the host-side hybridSplitKernel but runs on the GPU.
/// Falls back to the host computation if CUDA is not available.
pub fn launchHybridSplitKernel(num_missing: u32, fetch_frac_q16: u32) u32 {
    if (!build_options.has_cuda) {
        return hybridSplitKernel(num_missing, fetch_frac_q16);
    }
    
    // Allocate device memory for the result (single u32)
    var d_result: ?*anyopaque = null;
    errdefer cuda_runtime.free(d_result.?);
    if (cuda_runtime.malloc(@sizeOf(u32))) |ptr| {
        d_result = ptr;
    } else |_| {
        // If allocation fails, fall back to host computation
        return hybridSplitKernel(num_missing, fetch_frac_q16);
    }

    // Launch the CUDA kernel via the launcher
    hybridSplitKernelLauncher(num_missing, fetch_frac_q16, @as(*u32, @ptrCast(@alignCast(d_result.?))));
    // Wait for kernel to complete and check for errors
    cuda_runtime.deviceSync() catch {
        cuda_runtime.free(d_result.?);
        return hybridSplitKernel(num_missing, fetch_frac_q16);
    };
    cuda_runtime.checkLaunch() catch {
        cuda_runtime.free(d_result.?);
        return hybridSplitKernel(num_missing, fetch_frac_q16);
    };

    // Copy the result from device to host
    var result: u32 = undefined;
    cuda_runtime.memcpy(&result, d_result.?, @sizeOf(u32), .device_to_host) catch {
        // If copy fails, fall back to host computation
        cuda_runtime.free(d_result.?);
        return hybridSplitKernel(num_missing, fetch_frac_q16);
    };

    // Free device memory
    cuda_runtime.free(d_result.?);
    return result;
}

const testing = std.testing;

test "computeFetchFraction with example values" {
    // Example from FreeToken docs: pcie=20 GB/s, cpu=60 GB/s
    // q* = 20/(20+60) = 0.25
    // Q16 value = 0.25 * 65536 = 16384
    const result = computeFetchFraction(60.0, 20.0);
    try testing.expectEqual(16384, result);
}

test "computeFetchFraction edge cases" {
    // Zero bandwidths
    try testing.expectEqual(0, computeFetchFraction(0.0, 0.0));
    // Only PCIe bandwidth
    try testing.expectEqual(65536, computeFetchFraction(0.0, 10.0));
    // Only CPU bandwidth
    try testing.expectEqual(0, computeFetchFraction(10.0, 0.0));
}

test "hybridSplitKernel basic functionality" {
    // Test with 100 missing experts, 25% fetch fraction (16384 Q16)
    // Expected: ~25 experts to fetch
    const result = hybridSplitKernel(100, 16384);
    // Should be around 25 (exact value depends on minimax calculation)
    try testing.expect(result >= 20 and result <= 30);
}

test "hybridSplitKernel edge cases" {
    // Fetch none
    try testing.expectEqual(0, hybridSplitKernel(100, 0));
    // Fetch all
    try testing.expectEqual(100, hybridSplitKernel(100, 65536));
    // Zero missing
    try testing.expectEqual(0, hybridSplitKernel(0, 32768));
}

test "hybridSplitKernel minimax property" {
    // For symmetric case (50% split), should favor balanced distribution
    const result = hybridSplitKernel(100, 32768);  // 50%
    // Should be close to 50
    try testing.expect(result >= 45 and result <= 55);
}

test "launchHybridSplitKernel matches host version" {
    // Test that the CUDA wrapper returns the same result as the host function
    const num_missing: u32 = 100;
    const fetch_frac_q16: u32 = 16384; // 25%
    const host_result = hybridSplitKernel(num_missing, fetch_frac_q16);
    const cuda_result = launchHybridSplitKernel(num_missing, fetch_frac_q16);
    try testing.expectEqual(host_result, cuda_result);
}

test "launchHybridSplitKernel edge cases" {
    // Fetch none
    try testing.expectEqual(0, launchHybridSplitKernel(100, 0));
    // Fetch all
    try testing.expectEqual(100, launchHybridSplitKernel(100, 65536));
    // Zero missing
    try testing.expectEqual(0, launchHybridSplitKernel(0, 32768));
}