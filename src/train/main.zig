//! RLT training CLI — entrena pesos del adapter RLT (base model congelado).
//!
//! Usage: zig build train -- --d 64 --steps 200 --lr 3e-4 --out rlt_trained.gguf
//!
//! MODO ACTUAL (v1, synthetic): hidden states e[t] aleatorios para verificar
//! end-to-end el pipeline train→export. La señal entrena W_gate/W_state a
//! minimizar ‖u‖² — solo valida que el gradiente fluya y el GGUF exporte.
//! NO produce pesos útiles: eso requiere el capturador de hidden states del
//! modelo real (fase 2 — ver TODO_RLT.md).
//!
//! Salida: sidecar GGUF (rlt.feedback_alpha>0 + blk.0.rlt.feedback_{gate,state})
//! que el engine carga vía loadRltWeights (hybrid_layer.zig).
const builtin = @import("builtin");
const std = @import("std");
const rlt = @import("rlt_layer");
const train_mod = @import("train");

pub fn main(init: std.process.Init) !void {
    // Trainer usa page_allocator directo (sin DebugAllocator) para evitar
    // el deadlock en deinit() que causa el hang en trainReal con d grande.
    const gpa = std.heap.page_allocator;

    const io = init.io;
    const args: std.process.Args = init.minimal.args;
    var args_it = std.process.Args.Iterator.initAllocator(args, gpa) catch
        return error.OutOfMemory;
    defer args_it.deinit();
    _ = args_it.next(); // argv[0]

    // Parse CLI args
    var d: usize = 64;
    var steps: u32 = 200;
    var lr: f32 = 3e-4;
    var alpha_init: f32 = 0.15;
    var out_path: []const u8 = "rlt_trained.gguf";
    var seed: u64 = 42;
    var real_mode = false;
    var rltcap_path: []const u8 = "";
    var logits_target_path: []const u8 = "";
    var vocab_size: usize = 0;

    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--d") or std.mem.eql(u8, arg, "--dim")) {
            d = try std.fmt.parseInt(usize, args_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--steps")) {
            steps = try std.fmt.parseInt(u32, args_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--lr")) {
            lr = try std.fmt.parseFloat(f32, args_it.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--alpha")) {
            alpha_init = try std.fmt.parseFloat(f32, args_it.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--out")) {
            out_path = args_it.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--seed")) {
            seed = try std.fmt.parseInt(u64, args_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--real")) {
            real_mode = true;
        } else if (std.mem.eql(u8, arg, "--rltcap")) {
            rltcap_path = args_it.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--logits-target")) {
            logits_target_path = args_it.next() orelse return error.MissingValue;
        } else if (std.mem.eql(u8, arg, "--vocab-size")) {
            vocab_size = try std.fmt.parseInt(usize, args_it.next() orelse return error.MissingValue, 10);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            std.debug.print(
                \\RLT Adapter Training (synthetic hidden states)
                \\
                \\Usage: zig-ai-train [options]
                \\
                \\Options:
                \\  --d <n>        Embedding dimension (default: 64)
                \\  --steps <n>    Training steps (default: 200)
                \\  --lr <f>       Learning rate (default: 3e-4)
                \\  --alpha <f>    Initial RLT alpha (default: 0.15)
                \\  --out <path>   Output GGUF path (default: rlt_trained.gguf)
                \\  --seed <n>     Random seed (default: 42)
                \\  --help, -h     Show this help
                \\
                \\The training uses synthetic random hidden states to verify the backward
                \\pass and optimizer work. Output is an RLT sidecar GGUF with trained weights.
                \\
            , .{});
            return;
        }
    }

    std.debug.print("RLT Training: d={d}, steps={d}, lr={d:.4}, alpha={d:.4}\n", .{ d, steps, lr, alpha_init });

    if (real_mode) {
        if (rltcap_path.len == 0) {
            std.debug.print("[!] --real requiere --rltcap <path>\n", .{});
            return;
        }
        if (logits_target_path.len == 0) {
            std.debug.print("[!] --real requiere --logits-target <path>\n", .{});
            return;
        }
        if (vocab_size == 0) {
            std.debug.print("[!] --real requiere --vocab-size <n>\n", .{});
            return;
        }
        std.debug.print("RLT Training (real): rltcap={s} logits={s} vocab={d} steps={d} lr={d:.4}\n", .{ rltcap_path, logits_target_path, vocab_size, steps, lr });
        const metrics = try train_mod.trainReal(gpa, .{
            .rltcap_path = rltcap_path,
            .logits_target_path = logits_target_path,
            .vocab_size = vocab_size,
            .out_path = out_path,
            .steps = steps,
            .lr = lr,
            .d = d,
            .alpha_init = alpha_init,
            .seed = seed,
        });
        std.debug.print("Done! loss={d:.4} alpha={d:.4} saved to {s}\n", .{ metrics.loss, metrics.alpha, out_path });
        return;
    }

    // ─── Initialize weights ───
    const n_gate = d * 2 * d;
    const n_state = d * d;

    const w_gate = try gpa.alloc(f32, n_gate);
    defer gpa.free(w_gate);
    const w_state = try gpa.alloc(f32, n_state);
    defer gpa.free(w_state);

    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    const scale_gate = @sqrt(2.0 / @as(f32, @floatFromInt(3 * d)));
    const scale_state = @sqrt(2.0 / @as(f32, @floatFromInt(2 * d)));
    for (w_gate) |*v| v.* = rand.float(f32) * scale_gate * 2.0 - scale_gate;
    for (w_state) |*v| v.* = rand.float(f32) * scale_state * 2.0 - scale_state;

    var weights = rlt.RltWeights{
        .w_gate = w_gate,
        .w_state = w_state,
        .alpha = alpha_init,
    };

    // ─── Buffers ───
    var buf = try rlt.RltBuffers.alloc(gpa, .{ .d = d });
    defer buf.deinit(gpa);

    const adam = rlt.AdamWConfig{
        .lr = lr,
        .weight_decay = 0.01,
        .grad_clip = 1.0,
    };

    const e_buf = try gpa.alloc(f32, d);
    defer gpa.free(e_buf);
    const s_buf = try gpa.alloc(f32, d);
    defer gpa.free(s_buf);
    const d_out = try gpa.alloc(f32, d);
    defer gpa.free(d_out);
    @memset(s_buf, 0);

    var loss_ema: f32 = 0;

    // ─── Training loop ───
    var ts_start: std.posix.timespec = undefined;
    if (builtin.target.os.tag != .windows) _ = std.posix.system.clock_gettime(.MONOTONIC, &ts_start);

    for (0..steps) |step| {
        // Generate synthetic e[t]
        for (e_buf) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

        // Forward
        const u = rlt.forward(e_buf, s_buf, &weights, &buf);

        // Loss: ||u||² (energy-based objective)
        var loss: f32 = 0;
        for (u) |v| loss += v * v;
        loss /= @as(f32, @floatFromInt(d));

        // Backward: dL/du = 2*u/d
        for (0..d) |i| d_out[i] = 2.0 * u[i] / @as(f32, @floatFromInt(d));
        rlt.backward(d_out, &weights, &buf);

        // AdamW step
        rlt.adamwStep(w_gate, buf.dw_gate, buf.m_w_gate, buf.v_w_gate, adam, @intCast(step + 1), n_gate);
        rlt.adamwStep(w_state, buf.dw_state, buf.m_w_state, buf.v_w_state, adam, @intCast(step + 1), n_state);

        // EMA
        loss_ema = 0.95 * loss_ema + 0.05 * loss;

        // Update state
        @memcpy(s_buf, u);

        if (step % 20 == 0 or step == steps - 1) {
            std.debug.print("[{d:4}/{d}] loss={d:.6} alpha={d:.6}\n", .{ step + 1, steps, loss_ema, weights.alpha });
        }
    }

    var ts_end: std.posix.timespec = undefined;
    if (builtin.target.os.tag != .windows) _ = std.posix.system.clock_gettime(.MONOTONIC, &ts_end);
    const elapsed_ns = @as(u64, @intCast(ts_end.sec - ts_start.sec)) * 1_000_000_000 +| @as(u64, @intCast(ts_end.nsec -| ts_start.nsec));
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    std.debug.print("Training complete in {d:.1}s ({d:.1} steps/s)\n", .{ elapsed_s, @as(f64, @floatFromInt(steps)) / elapsed_s });

    // ─── Export GGUF (sidecar legible por el engine: keys blk.N.rlt.*) ───
    const export_mod = @import("export_gguf");
    var layer_weights = try gpa.alloc(export_mod.LayerWeights, 1);
    defer gpa.free(layer_weights);
    layer_weights[0] = .{ .w_gate = w_gate, .w_state = w_state };

    try export_mod.writeRltGguf(io, gpa, out_path, .{
        .num_layers = 1,
        .d = d,
        .alpha = weights.alpha,
    }, layer_weights);

    std.debug.print("Done! Trained weights saved to: {s}\n", .{out_path});
}
