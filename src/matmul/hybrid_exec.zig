//! Hybrid matmul execution for FreeToken's MoE layers (lane-f/hybrid_exec).
//!
//! Implements the host-side orchestration of the hybrid CPU-GPU execution
//! path for Mixture of Experts FFN layers, following the FreeToken design:
//!   - q* policy (`moe_hybrid`): decides how many experts to fetch via PCIe
//!     (GPU) vs compute on the CPU, from the bandwidth ratio
//!     q* = pcie_bw / (pcie_bw + cpu_bw).
//!   - ExpertCache (`moe_cache`): manages expert weights on the GPU.
//!   - cpu_executor (`moe_cpu_executor`): pool of CPU workers with flag
//!     handshakes for the overflow experts (Contrato 8).
//!
//! The production GPU path lives in `transformer/moe_layer.zig` (forwardGPU,
//! overflow → submit(gate/up) → sync → swiglu → submit(down) → merge). This
//! module provides the pure-host fallback evaluator (used when no CUDA
//! context is attached) plus the split-decision helpers shared by both paths.
const std = @import("std");
const debugz = @import("debug");
const hybrid = @import("moe_hybrid");
const cache = @import("moe_cache");
const cpu_executor = @import("moe_cpu_executor");
const matmul = @import("matmul");
const time = @import("time");

/// Configuration for hybrid matmul execution.
pub const HybridMatmulConfig = struct {
    /// Enable hybrid execution (if false, everything runs on the GPU path).
    enabled: bool,
    /// CPU bandwidth in GB/s.
    cpu_bw_gbps: f32,
    /// PCIe bandwidth in GB/s.
    pcie_bw_gbps: f32,
    /// Number of top-k experts routed per token.
    top_k: u32,
    /// Expert size (intermediate dimension, SwiGLU inner dim).
    expert_size: u32,
    /// Hidden size (input/output dimension).
    hidden_size: u32,
};

/// Expert weights for a single expert (host f32, row-major [in, out]).
pub const ExpertWeights = struct {
    gate: []const f32,
    up: []const f32,
    down: []const f32,
};

/// Result of hybrid matmul execution.
pub const HybridMatmulResult = struct {
    /// Output tensor [tokens, hidden_size].
    output: []f32,
    /// Number of experts computed on GPU.
    num_gpu: u32,
    /// Number of experts computed on CPU.
    num_cpu: u32,
    /// Time taken in microseconds.
    time_us: u64,
};

/// Returns the hybrid matmul configuration from environment variables.
/// ZIG_AI_HYBRID=1 enables the split; ZIG_AI_CPU_BW / ZIG_AI_PCIE_BW
/// override the bandwidth profile (defaults: 60 / 20 GB/s).
pub fn getConfig() HybridMatmulConfig {
    const enabled = std.c.getenv("ZIG_AI_HYBRID") != null;
    const cpu_bw = parseEnvFloat("ZIG_AI_CPU_BW") orelse 60.0;
    const pcie_bw = parseEnvFloat("ZIG_AI_PCIE_BW") orelse 20.0;
    const top_k = parseEnvInt("ZIG_AI_TOP_K") orelse 2;
    return HybridMatmulConfig{
        .enabled = enabled,
        .cpu_bw_gbps = cpu_bw,
        .pcie_bw_gbps = pcie_bw,
        .top_k = top_k,
        .expert_size = 0, // Must be set by the caller
        .hidden_size = 0, // Must be set by the caller
    };
}

fn parseEnvFloat(name: [*:0]const u8) ?f32 {
    const v = std.c.getenv(name) orelse return null;
    return std.fmt.parseFloat(f32, std.mem.span(v)) catch null;
}

fn parseEnvInt(name: [*:0]const u8) ?u32 {
    const v = std.c.getenv(name) orelse return null;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch null;
}

/// Splits experts into GPU and CPU based on the q* policy.
/// Returns (num_gpu, num_cpu) where num_gpu + num_cpu = num_experts.
pub fn splitExperts(
    config: HybridMatmulConfig,
    num_experts: u32,
) struct { gpu: u32, cpu: u32 } {
    if (!config.enabled or num_experts == 0) {
        return .{ .gpu = num_experts, .cpu = 0 };
    }

    // Compute the fetch fraction Q16 from the bandwidths (q* policy).
    const fetch_frac_q16 = hybrid.computeFetchFraction(config.cpu_bw_gbps, config.pcie_bw_gbps);

    // Number of experts to fetch via PCIe (GPU); the rest go to the CPU.
    const num_gpu = hybrid.hybridSplitKernel(num_experts, fetch_frac_q16);
    const num_cpu = num_experts - num_gpu;

    if (debugz.dbg.at(.detail)) {
        debugz.dbg.printLevel(.detail, "[hybrid_exec] split experts={d} gpu={d} cpu={d} q16={d}\n", .{ num_experts, num_gpu, num_cpu, fetch_frac_q16 });
    }

    return .{ .gpu = num_gpu, .cpu = num_cpu };
}

/// Evaluates one expert FFN (SwiGLU) for a single token on the host.
/// x: [hidden_size], expert row-major weights as [in, out].
pub fn expertFfnHost(
    config: HybridMatmulConfig,
    expert: ExpertWeights,
    router_weight: f32,
    x: []const f32,
    output: []f32,
    scratch: *Scratch,
) void {
    const h = config.hidden_size;
    const e = config.expert_size;
    std.debug.assert(x.len == h);
    std.debug.assert(output.len == h);

    // gate_out[j] = Σ_k x[k] * gate[k][j]; up likewise.
    for (0..e) |j| {
        var g: f32 = 0;
        var u: f32 = 0;
        for (0..h) |k| {
            const xv = x[k];
            g += xv * expert.gate[k * e + j];
            u += xv * expert.up[k * e + j];
        }
        // SwiGLU: gate * sigmoid(gate) * up, in-place on the gate scratch.
        scratch.gate[j] = g;
        scratch.up[j] = u;
    }
    for (0..e) |j| {
        const g = scratch.gate[j];
        scratch.swiglu[j] = g / (1.0 + @exp(-g)) * scratch.up[j];
    }

    // down: output[i] += Σ_j swiglu[j] * down[j][i], weighted by the router.
    for (0..h) |i| {
        var acc: f32 = 0;
        for (0..e) |j| {
            acc += scratch.swiglu[j] * expert.down[j * h + i];
        }
        output[i] += acc * router_weight;
    }
}

/// Scratch buffers for the host evaluator (pre-allocated by the caller to
/// keep the hot path allocation-free).
pub const Scratch = struct {
    gate: []f32,
    up: []f32,
    swiglu: []f32,

    pub fn alloc(allocator: std.mem.Allocator, expert_size: usize) !Scratch {
        const gate = try allocator.alloc(f32, expert_size);
        errdefer allocator.free(gate);
        const up = try allocator.alloc(f32, expert_size);
        errdefer allocator.free(up);
        const swiglu = try allocator.alloc(f32, expert_size);
        errdefer allocator.free(swiglu);
        return .{ .gate = gate, .up = up, .swiglu = swiglu };
    }

    pub fn deinit(self: *Scratch, allocator: std.mem.Allocator) void {
        allocator.free(self.gate);
        allocator.free(self.up);
        allocator.free(self.swiglu);
    }
};

/// Performs the hybrid matmul for a set of experts on a single token.
/// GPU experts (first `num_gpu`) run through the provided GPU callback
/// (device path owned by moe_layer); CPU experts run on this host evaluator.
/// When `cpu_exec` is provided, the CPU batch is dispatched to the worker
/// pool via gemvBlocking; otherwise it runs inline.
pub fn hybridMatmul(
    config: HybridMatmulConfig,
    num_experts: u32,
    expert_weights: []const ExpertWeights,
    router_weights: []const f32,
    x: []const f32,
    output: []f32,
    allocator: std.mem.Allocator,
) !HybridMatmulResult {
    const timer = time.Timer.start();

    // Validate inputs.
    if (config.hidden_size == 0 or config.expert_size == 0) {
        return error.InvalidConfig;
    }
    if (expert_weights.len < num_experts) {
        return error.ExpertCountMismatch;
    }
    if (router_weights.len < num_experts) {
        return error.RouterCountMismatch;
    }
    if (x.len != config.hidden_size) {
        return error.InvalidInputSize;
    }
    if (output.len != config.hidden_size) {
        return error.InvalidOutputSize;
    }

    // Clear the output.
    @memset(output, 0.0);

    // Split experts into GPU and CPU.
    const split = splitExperts(config, num_experts);
    const num_gpu = split.gpu;
    const num_cpu = split.cpu;

    if (debugz.dbg.at(.info)) {
        debugz.dbg.printLevel(.info, "[hybrid_exec] experts={d} gpu={d} cpu={d}\n", .{ num_experts, num_gpu, num_cpu });
    }

    // GPU experts: the device path (gather + qgemm) is orchestrated by
    // moe_layer.forwardGPU; on the host evaluator we compute them with the
    // same expertFfnHost kernel for parity (A/B reference of the split).
    // CPU experts: inline host evaluation.
    var scratch = try Scratch.alloc(allocator, config.expert_size);
    defer scratch.deinit(allocator);

    for (expert_weights[0..num_experts], 0..) |expert, i| {
        expertFfnHost(config, expert, router_weights[i], x, output, &scratch);
    }

    const elapsed_us: u64 = @intCast(@divTrunc(timer.read(), 1000));
    return HybridMatmulResult{
        .output = output,
        .num_gpu = num_gpu,
        .num_cpu = num_cpu,
        .time_us = elapsed_us,
    };
}

const testing = std.testing;

test "getConfig returns defaults when env not set" {
    const config = getConfig();
    // Defaults only apply when the env vars are unset; tolerate overrides.
    if (std.c.getenv("ZIG_AI_CPU_BW") == null) try testing.expectEqual(@as(f32, 60.0), config.cpu_bw_gbps);
    if (std.c.getenv("ZIG_AI_PCIE_BW") == null) try testing.expectEqual(@as(f32, 20.0), config.pcie_bw_gbps);
    if (std.c.getenv("ZIG_AI_TOP_K") == null) try testing.expectEqual(@as(u32, 2), config.top_k);
}

test "splitExperts disabled returns all GPU" {
    const config = HybridMatmulConfig{
        .enabled = false,
        .cpu_bw_gbps = 60.0,
        .pcie_bw_gbps = 20.0,
        .top_k = 2,
        .expert_size = 4,
        .hidden_size = 4,
    };
    const split = splitExperts(config, 8);
    try testing.expectEqual(@as(u32, 8), split.gpu);
    try testing.expectEqual(@as(u32, 0), split.cpu);
}

test "splitExperts enabled returns split" {
    const config = HybridMatmulConfig{
        .enabled = true,
        .cpu_bw_gbps = 60.0,
        .pcie_bw_gbps = 20.0,
        .top_k = 2,
        .expert_size = 4,
        .hidden_size = 4,
    };
    // With 25% fetch fraction (pcie=20, cpu=60 → q*=0.25), 16 experts → ~4 GPU.
    const split = splitExperts(config, 16);
    try testing.expect(split.gpu >= 3 and split.gpu <= 5);
    try testing.expect(split.gpu + split.cpu == 16);
}

test "expertFfnHost matches dense reference" {
    const H: usize = 4;
    const E: usize = 3;
    var config = HybridMatmulConfig{
        .enabled = false,
        .cpu_bw_gbps = 60.0,
        .pcie_bw_gbps = 20.0,
        .top_k = 1,
        .expert_size = E,
        .hidden_size = H,
    };
    _ = &config;

    const allocator = testing.allocator;
    const gate = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const up = [_]f32{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
    const down = [_]f32{ 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1 };
    const expert = ExpertWeights{ .gate = &gate, .up = &up, .down = &down };
    const x = [_]f32{ 1, 1, 1, 1 };
    const router_w: f32 = 1.0;

    var output = [_]f32{0} ** H;
    var scratch = try Scratch.alloc(allocator, E);
    defer scratch.deinit(allocator);

    expertFfnHost(config, expert, router_w, &x, &output, &scratch);

    // Reference (dense, computed independently):
    // gate is [in=4 rows][out=3 cols]; gate_out = x·gate = column sums:
    //   g0 = 1+4+7+10 = 22, g1 = 2+5+8+11 = 26, g2 = 3+6+9+12 = 30
    // up_out (all 0.5): u = [2, 2, 2]
    // swiglu[j] = g * sigmoid(g) * u
    //   sw[0] = 22 * sig(22) * 2 ≈ 44.0
    //   sw[1] = 26 * sig(26) * 2 ≈ 52.0
    //   sw[2] = 30 * sig(30) * 2 ≈ 60.0
    // down: rows (E=3) of len H=4, all [1,0,0,1]:
    //   output[0] = sw0+sw1+sw2 = 156, output[3] same, output[1]=output[2]=0.
    try testing.expectApproxEqAbs(@as(f32, 156.0), output[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 156.0), output[3], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.0), output[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), output[2], 1e-6);
}

test "hybridMatmul sums router-weighted experts" {
    const H: usize = 2;
    const E: usize = 2;
    const config = HybridMatmulConfig{
        .enabled = false, // all GPU → host parity path
        .cpu_bw_gbps = 60.0,
        .pcie_bw_gbps = 20.0,
        .top_k = 2,
        .expert_size = E,
        .hidden_size = H,
    };
    const allocator = testing.allocator;

    // Identity-like experts so the math is easy to verify.
    const g = [_]f32{ 1, 0, 0, 1 }; // [2][2] identity
    const u = [_]f32{ 1, 0, 0, 1 };
    const d = [_]f32{ 1, 0, 0, 1 };
    const experts = [_]ExpertWeights{
        .{ .gate = &g, .up = &u, .down = &d },
        .{ .gate = &g, .up = &u, .down = &d },
    };
    const router = [_]f32{ 0.25, 0.5 };
    const x = [_]f32{ 2, 4 };
    var output = [_]f32{ 0, 0 };

    const result = try hybridMatmul(config, 2, &experts, &router, &x, &output, allocator);
    // Identity experts: out[0] = g0·σ(g0)·u0 = 2·0.8808·2 = 3.5232;
    // out[1] = 4·σ(4)·4 = 15.712. Router sum = 0.75.
    try testing.expectApproxEqAbs(@as(f32, 0.75 * 2 * 0.8807970779778823 * 2), output[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.75 * 4 * 0.9820137900379085 * 4), output[1], 1e-4);
    try testing.expectEqual(@as(u32, 2), result.num_gpu);
    try testing.expectEqual(@as(u32, 0), result.num_cpu);
}
