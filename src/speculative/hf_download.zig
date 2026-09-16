//! C6-infra — descarga de sidecars DFlash/DSpark/DFlash2 desde Hugging Face.
//! Espejo de llama.cpp (--download-dflash/-dspark/-dflash2 + --model-draft):
//! el sidecar es un GGUF sibling en el MISMO repo que el target, con nombre
//! `<stem>-dflash|-dspark|-dflash2.gguf`. Reanudable (`curl -L -C -`) y sin
//! dependencias nuevas (curl vía std.process.Child; regla ola 2).
const std = @import("std");
const debugz = @import("debug");

pub const Variant = enum {
    dflash,
    dspark,
    dflash2,

    /// Sufijo del sibling según convención llama.cpp.
    pub fn suffix(self: Variant) [:0]const u8 {
        return switch (self) {
            .dflash => "-dflash",
            .dspark => "-dspark",
            .dflash2 => "-dflash2",
        };
    }
};

/// Referencia HF parseada: repo ("usuario/repo"), file ("dir/archivo.gguf")
/// y revisión (branch/tag/commit; default "main").
pub const HfRef = struct {
    repo: []const u8,
    file: []const u8,
    rev: []const u8 = "main",
};

/// Acepta las formas de llama.cpp para -m/--model-draft:
///   hf.co/<repo>/<file>[@rev]
///   https://huggingface.co/<repo>/<file>[@rev]
/// Devuelve null si `ref` es una ruta local (no URL).
pub fn parseHfRef(ref: []const u8) ?HfRef {
    var rest: []const u8 = ref;
    if (std.mem.startsWith(u8, rest, "https://huggingface.co/")) {
        rest = rest["https://huggingface.co/".len..];
    } else if (std.mem.startsWith(u8, rest, "http://huggingface.co/")) {
        // http lo redirige curl; normalizamos el parseo igualmente
        rest = rest["http://huggingface.co/".len..];
    } else if (std.mem.startsWith(u8, rest, "hf.co/")) {
        rest = rest["hf.co/".len..];
    } else if (std.mem.startsWith(u8, rest, "huggingface.co/")) {
        rest = rest["huggingface.co/".len..];
    } else {
        return null;
    }
    if (rest.len == 0) return null;
    // separa @rev del ÚLTIMO segmento (el archivo)
    const slash = std.mem.lastIndexOfScalar(u8, rest, '/') orelse return null;
    var file = rest[slash + 1 ..];
    var rev: []const u8 = "main";
    if (std.mem.lastIndexOfScalar(u8, file, '@')) |at| {
        rev = file[at + 1 ..];
        file = file[0..at];
    }
    return .{ .repo = rest[0..slash], .file = file, .rev = rev };
}

/// Nombre del sidecar para un archivo target: inserta el sufijo antes de la
/// extensión (.gguf); sin extensión conocida se añade al final.
pub fn sidecarFileName(allocator: std.mem.Allocator, target_file: []const u8, variant: Variant) ![]u8 {
    const suf = variant.suffix();
    if (std.mem.endsWith(u8, target_file, ".gguf")) {
        const base = target_file[0 .. target_file.len - ".gguf".len];
        return std.fmt.allocPrint(allocator, "{s}{s}.gguf", .{ base, suf });
    }
    return std.fmt.allocPrint(allocator, "{s}{s}", .{ target_file, suf });
}

/// Cache de sidecars: $XDG_CACHE_HOME/zig-ai/hf o $HOME/.cache/zig-ai/hf.
pub fn cacheDir(allocator: std.mem.Allocator) ![]u8 {
    if (std.c.getenv("XDG_CACHE_HOME")) |x| {
        const xd = std.mem.span(x);
        if (xd.len > 0) return std.fmt.allocPrint(allocator, "{s}/zig-ai/hf", .{xd});
    }
    const home_env = std.c.getenv("HOME") orelse return error.HomeUnknown;
    return std.fmt.allocPrint(allocator, "{s}/.cache/zig-ai/hf", .{std.mem.span(home_env)});
}

fn fileExists(io: std.Io, path: []const u8) bool {
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

/// mkdir -p vía libc (ya enlazada; Io.Dir no expone makePath).
fn mkdirP(path: []const u8) void {
    var buf: [512]u8 = undefined;
    if (path.len >= buf.len) return;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    var i: usize = 1;
    while (i <= path.len) : (i += 1) {
        if (buf[i] == '/' or buf[i] == 0) {
            buf[i] = 0;
            _ = std.c.mkdir(@ptrCast(buf[0..].ptr), 0o755);
            if (i < path.len) buf[i] = '/';
        }
    }
}

/// Resuelve el path LOCAL del sidecar para `target` (path local o ref HF):
/// 1. target local + sibling existente → usarlo tal cual (sin descargar).
/// 2. target HF-ref → curl -L -C - del sibling al cache (reanudable).
/// El caller es dueño del string devuelto.
pub fn ensureSidecar(io: std.Io, allocator: std.mem.Allocator, target: []const u8, variant: Variant) ![]u8 {
    if (parseHfRef(target)) |ref| {
        const fname = try sidecarFileName(allocator, ref.file, variant);
        defer allocator.free(fname);
        const dst_dir = try std.fmt.allocPrint(allocator, "{s}/{s}@{s}", .{ try cacheDir(allocator), ref.repo, ref.rev });
        const dst = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dst_dir, fname });
        const url = try std.fmt.allocPrint(allocator, "https://huggingface.co/{s}/resolve/{s}/{s}", .{ ref.repo, ref.rev, fname });
        defer {
            allocator.free(dst_dir);
            allocator.free(dst);
            allocator.free(url);
        }
        if (fileExists(io, dst)) return allocator.dupe(u8, dst);
        mkdirP(dst_dir);
        try curlDownload(io, allocator, url, dst);
        return allocator.dupe(u8, dst);
    }

    // Target LOCAL: buscar sibling junto al archivo.
    const dir = std.fs.path.dirname(target) orelse ".";
    const base = std.fs.path.basename(target);
    const sib_name = try sidecarFileName(allocator, base, variant);
    defer allocator.free(sib_name);
    const sib = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir, sib_name });
    if (fileExists(io, sib)) return sib;
    allocator.free(sib);

    // Último recurso: ya apunta directamente a un sidecar existente.
    if (fileExists(io, target)) return allocator.dupe(u8, target);
    return error.SidecarNotFound;
}

/// curl reanudable (-C -) con retry corto; propaga stderr recortado en el
/// mensaje de log gated (DEBUG). Error único DownloadFailed para CLI limpio.
fn curlDownload(io: std.Io, allocator: std.mem.Allocator, url: []const u8, dst: []const u8) !void {
    const argv = [_][]const u8{
        "curl",    "-L", "-C",            "-",  "--fail",
        "--retry", "3",  "--create-dirs", "-o", dst,
        url,
    };
    const result = std.process.run(allocator, io, .{
        .argv = &argv,
        .stdout_limit = .limited(16 * 1024),
        .stderr_limit = .limited(16 * 1024),
    }) catch return error.CurlSpawnFailed;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    const ok = switch (result.term) {
        .exited => |code| code == 0,
        else => false,
    };
    if (!ok) {
        debugz.dbg.print("[spec] curl falló para {s}: {s}\n", .{ url, result.stderr[0..@min(result.stderr.len, 512)] });
        return error.DownloadFailed;
    }
}

test "parseHfRef formas canónicas" {
    const r1 = parseHfRef("hf.co/user/repo/model-Q8_K_XL.gguf").?;
    try std.testing.expectEqualStrings("user/repo", r1.repo);
    try std.testing.expectEqualStrings("model-Q8_K_XL.gguf", r1.file);
    try std.testing.expectEqualStrings("main", r1.rev);
    const r2 = parseHfRef("https://huggingface.co/org/r/model.gguf@dev").?;
    try std.testing.expectEqualStrings("org/r", r2.repo);
    try std.testing.expectEqualStrings("dev", r2.rev);
    try std.testing.expect(parseHfRef("/ai/models/foo.gguf") == null);
}

test "sidecarFileName inserta sufijo antes de .gguf" {
    const a = try sidecarFileName(std.testing.allocator, "m-Q8_K_XL.gguf", .dflash);
    defer std.testing.allocator.free(a);
    try std.testing.expectEqualStrings("m-Q8_K_XL-dflash.gguf", a);
    const b = try sidecarFileName(std.testing.allocator, "sinext", .dspark);
    defer std.testing.allocator.free(b);
    try std.testing.expectEqualStrings("sinext-dspark", b);
}
