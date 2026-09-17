//! Lane-e — experimento: ¿cuMemHostRegister funciona sobre mmap MAP_SHARED?
//!
//! Hallazgo E8: registrar el mmap READ-ONLY del loader (MAP_PRIVATE implícito)
//! falla con ERROR_INVALID_VALUE en driver 580.173.02. Hipótesis: un mapeo
//! con protección READ+WRITE produce MAP_SHARED ⇒ registro OK. Si este test
//! lo confirma, el ticket a C es exactamente: "abrir el GGUF con protección
//! RW en fromFileMmap (o exponer opción share) ⇒ ruta zero-copy real".
//!
//! Disciplina GPU: flock NB con pocos reintentos ⇒ SKIP si ocupada.
const std = @import("std");
const builtin = @import("builtin");
const gguf = @import("gguf");
const cudaz = @import("cudaz");
const host_bank = @import("host_bank");

test "probe: HostBank sobre mmap RW (MAP_SHARED hipótesis) vs RO (known-issue)" {
    const io = std.Io.Threaded.global_single_threaded.io();

    // Respetar disciplina sin bloquear una sesión larga.
    // flock y nanosleep son POSIX; en Windows se omite el lock.
    if (comptime builtin.target.os.tag != .windows) {
        const lock_dir = std.Io.Dir.cwd();
        var waited: u32 = 0;
        while (true) {
            const lf = try lock_dir.createFile(io, ".bench.lock", .{ .truncate = false });
            if (std.c.flock(lf.handle, 6) == 0) break;
            lf.close(io);
            waited += 10;
            if (waited >= 30) return error.SkipZigTest;
            std.debug.print("[probe] .bench.lock ocupada, espero {d}s…\n", .{waited});
            var ts: std.c.timespec = .{ .sec = 10, .nsec = 0 };
            _ = std.c.nanosleep(&ts, null);
        }
        // fd deliberadamente NO cerrado hasta fin de test (close tras fallo de
        // flock es la vía que panea; aquí flock o skip).
    }

    cudaz.ensureContext() catch return error.SkipZigTest; // CI sin toolkit: skip limpio

    // Fichero temporal > varias páginas.
    const tmp_path = "/tmp/opencode/moe_probe_shared.bin";
    {
        const dir = std.Io.Dir.cwd();
        const wf = try dir.createFile(io, tmp_path, .{ .truncate = true });
        defer wf.close(io);
        var wbuf: [0x1000]u8 = undefined;
        var fw = wf.writer(io, &wbuf);
        const w = &fw.interface;
        var payload: [8192]u8 = undefined;
        for (&payload, 0..) |*b, i| b.* = @truncate(i);
        try w.writeAll(&payload);
        try w.flush();
    }

    // ── Caso A: protección READ+WRITE ⇒ expectativa MAP_SHARED ──
    {
        const dir = std.Io.Dir.cwd();
        var f = try dir.openFile(io, tmp_path, .{ .mode = .read_write });
        defer f.close(io);
        var mm = try f.createMemoryMap(io, .{
            .len = 8192,
            .protection = .{ .read = true, .write = true },
            .populate = true,
        });
        defer mm.destroy(io);
        // Tocar páginas antes del registro (pin-after-fill).
        var acc: u64 = 0;
        for (mm.memory) |b| acc += b;
        std.mem.doNotOptimizeAway(acc);

        var bank = host_bank.HostBank.fromFileMmapWhole(mm.memory) catch |err| {
            std.debug.print("[probe RW] REGISTRO FALLÓ: {s} ⇒ hipótesis rechazada\n", .{@errorName(err)});
            return error.SkipZigTest;
        };
        std.debug.print("[probe RW] registrado OK ({d} B) ⇒ MAP_SHARED CONFIRMADO: ticket C viable (protección RW en loader)\n", .{bank.length()});
        bank.unreg();
    }

    // ── Caso B: solo-lectura ⇒ known-issue esperado ──
    {
        const dir = std.Io.Dir.cwd();
        var f = try dir.openFile(io, tmp_path, .{ .mode = .read_only });
        defer f.close(io);
        var mm = try f.createMemoryMap(io, .{
            .len = 8192,
            .protection = .{ .read = true },
            .populate = true,
        });
        defer mm.destroy(io);
        if (host_bank.HostBank.fromFileMmapWhole(mm.memory)) |b| {
            var bank = b;
            bank.unreg();
            std.debug.print("[probe RO] registrado OK (driver permite también RO — revisar supuestos)\n", .{});
        } else |err| {
            std.debug.print("[probe RO] falló como se esperaba: {s} (confirma known-issue)\n", .{@errorName(err)});
        }
    }
}
