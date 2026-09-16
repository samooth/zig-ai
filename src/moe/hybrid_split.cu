//! Hybrid split kernel for FreeToken's q* policy.
//! Computes the optimal number of experts to fetch via PCIe given the number of missing experts
//! and the fetch fraction (Q16 fixed-point).
//! This kernel mirrors the logic in FreeToken's offload_kernels.py:345-354.
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void hybridSplitKernel(
    uint32_t num_missing,
    uint32_t fetch_frac_q16,
    uint32_t* result
) {
    // Handle edge cases
    if (num_missing == 0) {
        *result = 0;
        return;
    }
    if (fetch_frac_q16 == 0) {
        *result = 0;
        return;
    }
    if (fetch_frac_q16 == 65536) {
        *result = num_missing;
        return;
    }

    // Calculate number of experts to fetch via PCIe
    const uint32_t lo = (num_missing * fetch_frac_q16) >> 16;
    
    // Minimax: choose integer that minimizes max(PCIe_time, CPU_time)
    // PCIe time proportional to fetch_frac_q16 * num_fetched
    // CPU time proportional to (0x10000 - fetch_frac_q16) * num_computed
    const uint32_t cost_lo = (lo * (0x10000U - fetch_frac_q16)) > ((num_missing - lo) * fetch_frac_q16) ?
                             (lo * (0x10000U - fetch_frac_q16)) : ((num_missing - lo) * fetch_frac_q16);
    const uint32_t cost_hi = ((lo + 1) * (0x10000U - fetch_frac_q16)) > ((num_missing - lo - 1) * fetch_frac_q16) ?
                             ((lo + 1) * (0x10000U - fetch_frac_q16)) : ((num_missing - lo - 1) * fetch_frac_q16);
    
    *result = (cost_lo <= cost_hi) ? lo : lo + 1;
}

// Launcher function for the hybridSplitKernel (C linkage: declared
// `extern "c" fn hybridSplitKernelLauncher` from src/moe/hybrid.zig).
extern "C" void hybridSplitKernelLauncher(
    uint32_t num_missing,
    uint32_t fetch_frac_q16,
    uint32_t* d_result
) {
    hybridSplitKernel<<<1, 1>>>(num_missing, fetch_frac_q16, d_result);
}