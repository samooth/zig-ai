//! Token injection — construye la secuencia mixta texto+imagen para el
//! LLM target: embeddings y position-ids M-RoPE 2D.
//!
//! Referencias (llama.cpp mtmd):
//!   - decode_embd_batch + set_position_mrope_2d: mtmd-helper.cpp:142-213
//!   - pos de imagen (MTMD_POS_TYPE_MROPE): mtmd.cpp:1920-1936
//!     pos.t = pos_0 (secuencial); pos.x = pos_0 + (i % nx);
//!     pos.y = pos_0 + (i / nx); z=0. Cada imagen consume max(nx, ny)
//!     posiciones (mtmd_image_tokens_get_n_pos, mtmd.cpp:1982-1984).
//!   - Texto: 4 ids iguales = posición secuencial (set_position_normal).
//!
//! Contract con el LLM zig-ai: los tokens de imagen NO pasan por la tabla
//! de embeddings del target — se inyectan como embeddings externos en las
//! posiciones donde el template pone <|image_pad|>/vision markers. La
//! Fase 3 conectará esto con runHybridInference (los pos-ids 2D requieren
//! extensión del path RoPE del target: la applyRoPEMultiSection actual
//! solo acepta start_pos escalar).
const std = @import("std");
const debugz = @import("debug");

pub const ImgGrid = struct { nx: usize, ny: usize };

pub const InjectError = error{
    ShapeMismatch,
    OutOfMemory,
    NoImageMarkers,
};

pub const InjectedSequence = struct {
    /// Embeddings de TODA la secuencia (texto + imagen), [seq_total, hidden]
    embeddings: []f32,
    /// Posiciones totales consumidas (para el LLM: max avanza este nº)
    n_pos_total: usize,
    /// Pos-ids M-RoPE por embedding: 4 ids (t, h, w, e)
    pos_ids: [][4]i32,

    pub fn deinit(self: *InjectedSequence, allocator: std.mem.Allocator) void {
        allocator.free(self.embeddings);
        allocator.free(self.pos_ids);
        self.embeddings = &.{};
        self.pos_ids = &.{};
    }
};

pub const ImageSpan = struct {
    /// Índice del primer embedding de imagen en la secuencia final
    start: usize,
    /// Nº de embeddings de imagen (tokens de imagen)
    n_tokens: usize,
};

/// Combina embeddings de texto e imagen.
///
/// - `text_emb`: [n_text, hidden] — embeddings del texto YA generados por
///   el caller (lookup del target) CON el hueco de la imagen incluido o
///   no (ver `marker_token`).
/// - `image_emb`: [n_img, img_dim] — salida del projector (dim puede ser
///   != hidden si deepstack; el caller debe validar contra n_embd_inp).
/// - `marker_idx`: índice en la secuencia de texto donde insertar la
///   imagen (el template pone N tokens <|image_pad|>; sustituimos).
///   Si null: prepend al inicio.
/// - `img_grid`: (nx, ny) grid FINAL de la imagen (tras spatial merge).
///
/// Los pos-ids de imagen siguen MTMD_POS_TYPE_MROPE (mtmd.cpp:1925-1929):
///   t = posición secuencial base; x = t + (i % nx); y = t + (i / nx).
pub fn injectImageEmbeddings(
    allocator: std.mem.Allocator,
    text_emb: []const f32, // [n_text, hidden]
    n_text: usize,
    hidden: usize,
    image_emb: []const f32, // [n_img, hidden] (validado por el caller)
    n_img: usize,
    img_grid: ImgGrid,
    marker_token: ?[]const u32, // tokens marcadores en la seq de texto
    text_tokens: []const u32, // secuencia tokenizada del texto
    image_token_id: u32, // id de <|image_pad| (Qwen3.8: 248056)
) InjectError!InjectedSequence {
    // Localizar los n_img marcadores consecutivos... El template Qwen
    // expande UNA sola posición: los embeddings de imagen SUSTITUYEN a
    // los marcadores. Estrategia: el caller tokeniza el template con un
    // marcador; reemplazamos 1 marcador por n_img embeddings.
    var marker_idx: ?usize = null;
    if (marker_token != null) {
        for (text_tokens, 0..) |tok, i| {
            if (tok == image_token_id) {
                marker_idx = i;
                break;
            }
        }
        if (marker_idx == null) return InjectError.NoImageMarkers;
    }

    const ins_at = marker_idx orelse 0; // null ⇒ prepend
    // Marker: sustituye 1 token por n_img. Prepend: añade n_img sin quitar.
    const n_total = if (marker_idx != null) n_text - 1 + n_img else n_text + n_img;

    const emb = allocator.alloc(f32, n_total * hidden) catch
        return InjectError.OutOfMemory;
    errdefer allocator.free(emb);
    const pos = allocator.alloc([4]i32, n_total) catch {
        allocator.free(emb);
        return InjectError.OutOfMemory;
    };
    errdefer allocator.free(pos);

    // pos base de la imagen: la posición secuencial donde cae el primer
    // token de imagen (= ins_at en términos de secuencia final).
    const pos_0: i32 = @intCast(ins_at);

    var cursor: usize = 0;

    // Prepend (marker null): imagen PRIMERO, luego todo el texto.
    if (marker_idx == null) {
        for (0..n_img) |k| {
            const src = image_emb[k * hidden ..][0..hidden];
            @memcpy(emb[cursor * hidden ..][0..hidden], src);
            // MTMD_POS_TYPE_MROPE (mtmd.cpp:1925-1929)
            pos[cursor] = .{
                pos_0,
                pos_0 + @as(i32, @intCast(k % img_grid.nx)),
                pos_0 + @as(i32, @intCast(k / img_grid.nx)),
                0,
            };
            cursor += 1;
        }
        for (0..n_text) |t| {
            @memcpy(emb[cursor * hidden ..][0..hidden], text_emb[t * hidden ..][0..hidden]);
            const p: i32 = @intCast(cursor);
            pos[cursor] = .{ p, p, p, p };
            cursor += 1;
        }
        std.debug.assert(cursor == n_total);
    } else {
        // Marker: sustituir 1 token marcador por n_img embeddings.
        for (0..n_text) |t| {
            if (t == ins_at) {
                for (0..n_img) |k| {
                    const src = image_emb[k * hidden ..][0..hidden];
                    @memcpy(emb[cursor * hidden ..][0..hidden], src);
                    pos[cursor] = .{
                        pos_0,
                        pos_0 + @as(i32, @intCast(k % img_grid.nx)),
                        pos_0 + @as(i32, @intCast(k / img_grid.nx)),
                        0,
                    };
                    cursor += 1;
                }
                continue;
            }
            // Token de texto: embedding directo + pos secuencial
            @memcpy(emb[cursor * hidden ..][0..hidden], text_emb[t * hidden ..][0..hidden]);
            const p: i32 = @intCast(cursor);
            pos[cursor] = .{ p, p, p, p };
            cursor += 1;
        }
        std.debug.assert(cursor == n_total);
    }

    // Nº de posiciones consumidas: cada imagen consume max(nx, ny)
    // posiciones secuenciales del contexto (mtmd.cpp:1982-1984), +1 por
    // cada token de texto. Marker: (n_text-1) texto + max(nx,ny) imagen;
    // prepend: n_text + max(nx,ny).
    const img_pos = @max(img_grid.nx, img_grid.ny);
    const text_pos = if (marker_idx != null) n_text - 1 else n_text;

    return .{
        .embeddings = emb,
        .pos_ids = pos,
        .n_pos_total = text_pos + img_pos,
    };
}

/// Variante sin marcador: imagen PREPENDIDA al texto (uso simple con
/// prompt manual sin template).
pub fn injectPrepend(
    allocator: std.mem.Allocator,
    text_emb: []const f32,
    n_text: usize,
    hidden: usize,
    image_emb: []const f32,
    n_img: usize,
    img_grid: ImgGrid,
) InjectError!InjectedSequence {
    return injectImageEmbeddings(
        allocator,
        text_emb,
        n_text,
        hidden,
        image_emb,
        n_img,
        img_grid,
        null,
        &.{},
        0,
    );
}

// ── 10.7 VIDEO: expansión de markers por tipo + pos-ids M-RoPE ───────────────

/// Input de visión para la expansión (embeddings YA calculados por el
/// encoder: still o par temporal). Duplica el shape de VisionInput de
/// main.zig sin depender de él (main importa este módulo, no al revés).
pub const ExpandInput = struct {
    n_tokens: usize,
    grid_x: usize,
    grid_y: usize,
    /// índice de chunk temporal dentro de su vídeo; null = still
    video_chunk: ?usize = null,
    /// factor temporal vLLM (seconds-per-pair·tps); 0 ⇒ t constante
    /// (llama.cpp mtnd MROPE)
    t_factor: f32 = 0,
};

/// Span de un input en la secuencia expandida.
pub const ExpandSpan = struct {
    /// posición del primer token del input en `ids`
    start: usize,
    /// nº de tokens (embeddings) del input
    n: usize,
    /// índice del input en `inputs`
    img: usize,
};

pub const ExpandResult = struct {
    /// tokens expandidos (markers 1-token → N tokens marker_id)
    ids: []u32,
    /// pos-ids M-RoPE por token [len][4] (t, h, w, e); null si no hubo expansión
    pos_ids: [][4]i32,
    /// spans por input, en orden de aparición
    spans: []ExpandSpan,
    /// posiciones de contexto LLM consumidas por TODOS los inputs
    pos_consumed: usize,

    pub fn deinit(self: *ExpandResult, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.pos_ids);
        allocator.free(self.spans);
        self.ids = &.{};
        self.pos_ids = &.{};
        self.spans = &.{};
    }
};

pub const ExpandError = error{
    OutOfMemory,
    ShapeMismatch,
};

/// Expande los markers del prompt con los tokens de cada input y calcula
/// los pos-ids M-RoPE (llama.cpp MTMD_POS_TYPE_MROPE, mtmd.cpp:1922-1929).
///
/// Matching por TIPO (10.7): un marker de VÍDEO expande TODOS los chunks
/// consecutivos del siguiente vídeo pendiente (llama.cpp fusiona los pares
/// de un vídeo en un único chunk; vLLM trata la secuencia completa como
/// "video tokens"). Un marker de IMAGEN expande UNA imagen (1:1 por orden
/// de aparición). Los markers se ordenan por posición antes del matching.
///
/// Pos-ids (mtmd.cpp:1922-1929): por cada span, t = pos_0, x = pos_0 +
/// (k % nx), y = pos_0 + (k / nx), e = 0; el texto usa (p,p,p,p)
/// secuencial. Cada input consume max(nx, ny) posiciones de contexto
/// (mtmd.cpp:1982-1984). Para chunks de vídeo con t_factor > 0, el eje t
/// sigue vLLM (qwen2_5_vl.py:1310): t = video_pos_0 + chunk_idx·t_factor
/// (video_pos_0 = pos_0 del PRIMER chunk del vídeo).
///
/// - `prompt_ids`: tokens con markers 1-token ya presentes.
/// - `image_marker_idxs`/`video_marker_idxs`: índices de markers por tipo
///   (hints del encode; pueden solaparse en id con modelos text-only de
///   test — el tipo lo da la lista, no el id).
/// - `marker_id`: id con el que se rellenan los tokens expandidos (el
///   embedding se sobreescribe por el ViT; el id es cosmético salvo
///   debug).
/// Marcadores vision Qwen (mismo conjunto que main.zig — duplicado
/// deliberado: token_inject no importa main).
pub const vision_markers = [_][]const u8{
    "<|vision_start|>",
    "<|image_pad|>",
    "<|vision_end|>",
    "<|video_pad|>",
};

/// UX vision automática (10.7, patrón mtmd-cli.cpp:441-447): con inputs de
/// visión cargados y SIN marcadores en el prompt, ANTEPONE un bloque por
/// input en orden — still → image_pad, vídeo → video_pad (1 por VÍDEO: el
/// matching de expandVisionTokens expande todos sus chunks). Con marcadores
/// ya presentes devuelve copia del prompt tal cual (control manual manda).
/// `inputs`: metadatos de encodeVision (los mismos ExpandInput).
pub fn buildVisionPrompt(
    allocator: std.mem.Allocator,
    raw_prompt: []const u8,
    inputs: ?[]const ExpandInput,
) ![]const u8 {
    const ins = inputs orelse return try allocator.dupe(u8, raw_prompt);
    if (ins.len == 0) return try allocator.dupe(u8, raw_prompt);
    for (vision_markers) |m| {
        if (std.mem.indexOf(u8, raw_prompt, m) != null) {
            return try allocator.dupe(u8, raw_prompt);
        }
    }
    // contar stills y vídeos: un vídeo = secuencia de chunks con vc
    // ascendente (vc reinicia a 0 por vídeo); un DECREMENTO de vc marca el
    // inicio del vídeo siguiente (encodeVision numera chunks por vídeo)
    var n_still: usize = 0;
    var n_videos: usize = 0;
    var last_vc: ?usize = null;
    for (ins) |v| {
        if (v.video_chunk) |vc| {
            const new_video = last_vc == null or vc <= last_vc.?;
            if (new_video) n_videos += 1;
            last_vc = vc;
        } else {
            n_still += 1;
            last_vc = null;
        }
    }
    const blk_still = "<|vision_start|><|image_pad|><|vision_end|>";
    const blk_video = "<|vision_start|><|video_pad|><|vision_end|>";
    const total: usize = raw_prompt.len + (n_still * blk_still.len) + (n_videos * blk_video.len);
    const out = try allocator.alloc(u8, total);
    errdefer allocator.free(out);
    var off: usize = 0;
    for (0..n_still) |_| {
        @memcpy(out[off .. off + blk_still.len], blk_still);
        off += blk_still.len;
    }
    for (0..n_videos) |_| {
        @memcpy(out[off .. off + blk_video.len], blk_video);
        off += blk_video.len;
    }
    @memcpy(out[off .. off + raw_prompt.len], raw_prompt);
    return out;
}

pub fn expandVisionTokens(
    allocator: std.mem.Allocator,
    prompt_ids: []const u32,
    inputs: []const ExpandInput,
    image_marker_idxs: []const usize,
    video_marker_idxs: []const usize,
    marker_id: u32,
) ExpandError!ExpandResult {
    // markers por tipo, ordenados por aparición
    var marker_idxs: std.ArrayList(usize) = .empty;
    defer marker_idxs.deinit(allocator);
    try marker_idxs.appendSlice(allocator, image_marker_idxs);
    try marker_idxs.appendSlice(allocator, video_marker_idxs);
    if (marker_idxs.items.len > 0) {
        std.mem.sort(usize, marker_idxs.items, {}, std.sort.asc(usize));
    }

    var img_cursor: usize = 0; // siguiente input still no consumido
    var vid_cursor: usize = 0; // siguiente chunk de vídeo no consumido
    // por marker: lista de inputs que expande
    var assigned: std.ArrayList([]usize) = .empty;
    defer {
        for (assigned.items) |l| allocator.free(l);
        assigned.deinit(allocator);
    }
    var any_assigned = false;
    for (marker_idxs.items) |mi| {
        const is_video_marker = blk: {
            for (video_marker_idxs) |vh| {
                if (vh == mi) break :blk true;
            }
            break :blk false;
        };
        if (is_video_marker) {
            // saltar imágenes (y chunks de vídeos ya consumidos) hasta el
            // primer chunk de vídeo SIN USAR; los chunks de un vídeo son
            // contiguos por construcción de encodeVision
            var start = vid_cursor;
            while (start < inputs.len and inputs[start].video_chunk == null) : (start += 1) {}
            var endv = start;
            while (endv < inputs.len and inputs[endv].video_chunk != null) : (endv += 1) {}
            if (endv == start) continue; // sin chunks pendientes
            const lst = try allocator.alloc(usize, endv - start);
            for (start..endv, 0..) |ix, li| lst[li] = ix;
            try assigned.append(allocator, lst);
            vid_cursor = endv;
            any_assigned = true;
        } else {
            if (img_cursor < inputs.len and inputs[img_cursor].video_chunk == null) {
                const lst = try allocator.alloc(usize, 1);
                lst[0] = img_cursor;
                try assigned.append(allocator, lst);
                img_cursor += 1;
                any_assigned = true;
            }
        }
    }
    if (!any_assigned) {
        // sin matching: ids sin tocar, pos_ids lineal (texto puro)
        const ids = try allocator.dupe(u32, prompt_ids);
        errdefer allocator.free(ids);
        const pos = try allocator.alloc([4]i32, prompt_ids.len);
        errdefer allocator.free(pos);
        for (pos, 0..) |*p, i| p.* = .{ @intCast(i), @intCast(i), @intCast(i), @intCast(i) };
        const spans = try allocator.alloc(ExpandSpan, 0);
        return .{ .ids = ids, .pos_ids = pos, .spans = spans, .pos_consumed = 0 };
    }

    // ── Expansión: cada marker 1-token → N tokens marker_id
    var total_tokens = prompt_ids.len;
    for (assigned.items) |lst| {
        var add: usize = 0;
        for (lst) |ix| add += inputs[ix].n_tokens;
        total_tokens += add - 1;
    }
    const expanded = try allocator.alloc(u32, total_tokens);
    errdefer allocator.free(expanded);

    var spans: std.ArrayList(ExpandSpan) = .empty;
    errdefer spans.deinit(allocator);

    {
        var src_off: usize = 0;
        var dst_off: usize = 0;
        for (marker_idxs.items, 0..) |mi, mk| {
            if (mk >= assigned.items.len) break;
            const lst = assigned.items[mk];
            const pre_len = mi - src_off;
            @memcpy(expanded[dst_off .. dst_off + pre_len], prompt_ids[src_off .. src_off + pre_len]);
            dst_off += pre_len;
            src_off = mi + 1;
            for (lst) |ix| {
                const v = inputs[ix];
                @memset(expanded[dst_off .. dst_off + v.n_tokens], marker_id);
                try spans.append(allocator, .{ .start = dst_off, .n = v.n_tokens, .img = ix });
                dst_off += v.n_tokens;
            }
        }
        const tail_len = prompt_ids.len - src_off;
        @memcpy(expanded[dst_off .. dst_off + tail_len], prompt_ids[src_off..]);
    }

    // ── Pos-ids M-RoPE
    const pos = try allocator.alloc([4]i32, expanded.len);
    errdefer allocator.free(pos);
    var ctx: i32 = 0;
    var pos_consumed: usize = 0;
    var video_pos_0: i32 = 0; // pos_0 del primer chunk del vídeo en curso
    var si: usize = 0;
    var i: usize = 0;
    while (i < expanded.len) {
        if (si < spans.items.len and i == spans.items[si].start) {
            const sp = spans.items[si];
            const v = inputs[sp.img];
            const pos_0 = ctx;
            const span_len: i32 = @intCast(@max(v.grid_x, v.grid_y));
            // primer chunk del vídeo: fija la base del eje t (vLLM). Señal
            // de nuevo vídeo = vc del chunk anterior DECRECE (vc reinicia a 0
            // por vídeo; el chunk previo era still o de otro vídeo)
            const is_first_chunk = if (v.video_chunk) |vc|
                sp.img == 0 or inputs[sp.img - 1].video_chunk == null or inputs[sp.img - 1].video_chunk.? >= vc
            else
                false;
            if (is_first_chunk) video_pos_0 = pos_0;
            const t: i32 = if (v.video_chunk) |vc|
                (if (v.t_factor > 0)
                    video_pos_0 + @as(i32, @intFromFloat(@as(f32, @floatFromInt(vc)) * v.t_factor))
                else
                    pos_0)
            else
                pos_0;
            for (0..sp.n) |kk| {
                pos[i + kk] = .{
                    t,
                    pos_0 + @as(i32, @intCast(kk % v.grid_x)),
                    pos_0 + @as(i32, @intCast(kk / v.grid_x)),
                    0,
                };
            }
            i += sp.n;
            si += 1;
            ctx += span_len;
            pos_consumed += @intCast(span_len);
        } else {
            pos[i] = .{ ctx, ctx, ctx, ctx };
            ctx += 1;
            i += 1;
        }
    }

    return .{
        .ids = expanded,
        .pos_ids = pos,
        .spans = try spans.toOwnedSlice(allocator),
        .pos_consumed = pos_consumed,
    };
}

// ── Tests ──────────────────────────────────────────────────────────────────

const testing = std.testing;

test "injectPrepend: imagen 4 tokens + texto 3 → 7 embeds, pos 2D" {
    const a = testing.allocator;
    const hidden = 2;
    // texto: 3 tokens con emb = índice
    const text_emb = [_]f32{ 1, 1, 2, 2, 3, 3 };
    // imagen: 4 embeds (grid 2x2)
    const img_emb = [_]f32{ 9, 9, 9, 9, 8, 8, 8, 8 };

    var inj = try injectPrepend(a, &text_emb, 3, hidden, &img_emb, 4, .{ .nx = 2, .ny = 2 });
    defer inj.deinit(a);

    try testing.expectEqual(@as(usize, 7), inj.pos_ids.len);
    // Primer token de imagen: pos (0, 0, 0, 0)
    try testing.expectEqualSlices(i32, &.{ 0, 0, 0, 0 }, &inj.pos_ids[0]);
    // Token 1 (x=1): (0, 1, 0, 0)
    try testing.expectEqualSlices(i32, &.{ 0, 1, 0, 0 }, &inj.pos_ids[1]);
    // Token 2 (y=1): (0, 0, 1, 0)
    try testing.expectEqualSlices(i32, &.{ 0, 0, 1, 0 }, &inj.pos_ids[2]);
    // Token 3: (0, 1, 1, 0)
    try testing.expectEqualSlices(i32, &.{ 0, 1, 1, 0 }, &inj.pos_ids[3]);
    // Primer texto: pos (4,4,4,4)
    try testing.expectEqualSlices(i32, &.{ 4, 4, 4, 4 }, &inj.pos_ids[4]);
    // n_pos: 3 texto + max(2,2) imagen = 5
    try testing.expectEqual(@as(usize, 5), inj.n_pos_total);
    // Embeds de imagen primeros
    try testing.expectApproxEqAbs(9.0, inj.embeddings[0], 1e-6);
    try testing.expectApproxEqAbs(1.0, inj.embeddings[8], 1e-6); // texto 1
}

test "injectImageEmbeddings: marcador en medio se sustituye" {
    const a = testing.allocator;
    const hidden = 1;
    const text_emb = [_]f32{ 10, 20, 30 }; // 3 tokens
    const tokens = [_]u32{ 100, 248056, 102 }; // marcador en idx 1
    const img_emb = [_]f32{ 7, 7 }; // 2 tokens de imagen

    var inj = try injectImageEmbeddings(a, &text_emb, 3, hidden, &img_emb, 2, .{ .nx = 2, .ny = 1 }, &tokens, &tokens, 248056);
    defer inj.deinit(a);

    // total = 3-1+2 = 4
    try testing.expectEqual(@as(usize, 4), inj.pos_ids.len);
    // [10, 7, 7, 30]... no: [10, img0, img1, 30] con emb img=7
    try testing.expectApproxEqAbs(10.0, inj.embeddings[0], 1e-6);
    try testing.expectApproxEqAbs(7.0, inj.embeddings[1], 1e-6);
    try testing.expectApproxEqAbs(7.0, inj.embeddings[2], 1e-6);
    try testing.expectApproxEqAbs(30.0, inj.embeddings[3], 1e-6);
    // pos imagen base = 1 (idx del marcador)
    try testing.expectEqualSlices(i32, &.{ 1, 1, 1, 0 }, &inj.pos_ids[1]);
    try testing.expectEqualSlices(i32, &.{ 1, 2, 1, 0 }, &inj.pos_ids[2]); // x=1
    // texto después: pos 3
    try testing.expectEqualSlices(i32, &.{ 3, 3, 3, 3 }, &inj.pos_ids[3]);
}

// ── 10.7 F3: tests de regresión del matching por tipo + pos-ids ─────────────

test "expandVisionTokens: video_pad expande TODOS los chunks del vídeo" {
    const a = testing.allocator;
    // prompt: [vs, video_pad, ve, "D", "e"] (vs/ve son tokens normales aquí)
    const prompt = [_]u32{ 10, 999, 12, 20, 21 };
    // vídeo de 3 chunks (grid 2x2, 4 tok cada) + 1 still
    const inputs = [_]ExpandInput{
        .{ .n_tokens = 4, .grid_x = 2, .grid_y = 2, .video_chunk = 0, .t_factor = 0 },
        .{ .n_tokens = 4, .grid_x = 2, .grid_y = 2, .video_chunk = 1, .t_factor = 0 },
        .{ .n_tokens = 4, .grid_x = 2, .grid_y = 2, .video_chunk = 2, .t_factor = 0 },
    };
    var r = try expandVisionTokens(a, &prompt, &inputs, &.{}, &.{1}, 999);
    defer r.deinit(a);
    // 5 - 1 + 12 = 16 tokens
    try testing.expectEqual(@as(usize, 16), r.ids.len);
    try testing.expectEqual(@as(usize, 3), r.spans.len);
    // spans consecutivos: [1..5), [5..9), [9..13)
    try testing.expectEqual(@as(usize, 1), r.spans[0].start);
    try testing.expectEqual(@as(usize, 4), r.spans[0].n);
    try testing.expectEqual(@as(usize, 5), r.spans[1].start);
    try testing.expectEqual(@as(usize, 9), r.spans[2].start);
    // 3 chunks × max(2,2) = 6 pos de contexto (llama.cpp n_pos)
    try testing.expectEqual(@as(usize, 6), r.pos_consumed);
    // pos-ids chunk 0: t=1 (pos_0 texto=1), x=1+k%2, y=1+k/2
    try testing.expectEqualSlices(i32, &.{ 1, 1, 1, 0 }, &r.pos_ids[1]);
    try testing.expectEqualSlices(i32, &.{ 1, 2, 1, 0 }, &r.pos_ids[2]);
    try testing.expectEqualSlices(i32, &.{ 1, 1, 2, 0 }, &r.pos_ids[3]);
    // chunk 1: pos_0 = 1+2 = 3 (tras chunk 0 que consume 2)
    try testing.expectEqualSlices(i32, &.{ 3, 3, 3, 0 }, &r.pos_ids[5]);
    // chunk 2: pos_0 = 5
    try testing.expectEqualSlices(i32, &.{ 5, 5, 5, 0 }, &r.pos_ids[9]);
    // texto tras el vídeo: ctx = 6
    try testing.expectEqualSlices(i32, &.{ 6, 6, 6, 6 }, &r.pos_ids[13]);
}

test "expandVisionTokens: image_pad 1:1 con imágenes" {
    const a = testing.allocator;
    // prompt: [img_pad, "+", img_pad, "!"]
    const prompt = [_]u32{ 999, 5, 999, 7 };
    const inputs = [_]ExpandInput{
        .{ .n_tokens = 2, .grid_x = 2, .grid_y = 1 }, // still 0
        .{ .n_tokens = 2, .grid_x = 1, .grid_y = 2 }, // still 1
    };
    var r = try expandVisionTokens(a, &prompt, &inputs, &.{ 0, 2 }, &.{}, 999);
    defer r.deinit(a);
    try testing.expectEqual(@as(usize, 7), r.ids.len);
    try testing.expectEqual(@as(usize, 2), r.spans.len);
    // imagen 0 en [0..2), imagen 1 en [3..5)
    try testing.expectEqual(@as(usize, 0), r.spans[0].start);
    try testing.expectEqual(@as(usize, 3), r.spans[1].start);
    // pos: still0 t=0 (x=0..1), texto "+" pos 2, still1 pos_0=3 (y=3+k)
    try testing.expectEqualSlices(i32, &.{ 0, 0, 0, 0 }, &r.pos_ids[0]);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 0, 0 }, &r.pos_ids[1]);
    try testing.expectEqualSlices(i32, &.{ 2, 2, 2, 2 }, &r.pos_ids[2]);
    try testing.expectEqualSlices(i32, &.{ 3, 3, 3, 0 }, &r.pos_ids[3]);
    try testing.expectEqualSlices(i32, &.{ 3, 3, 4, 0 }, &r.pos_ids[4]);
    // still1 consume max(1,2)=2 ⇒ texto final pos 5
    try testing.expectEqualSlices(i32, &.{ 5, 5, 5, 5 }, &r.pos_ids[5]);
}

test "expandVisionTokens: mezcla imagen + vídeo (2 vídeos separados)" {
    const a = testing.allocator;
    // [img_pad, vid_pad, texto, vid_pad]
    const prompt = [_]u32{ 999, 998, 3, 998, 9 };
    const inputs = [_]ExpandInput{
        .{ .n_tokens = 2, .grid_x = 2, .grid_y = 1 }, // still
        .{ .n_tokens = 2, .grid_x = 2, .grid_y = 1, .video_chunk = 0 }, // vídeo A c0
        .{ .n_tokens = 2, .grid_x = 2, .grid_y = 1, .video_chunk = 1 }, // vídeo A c1
        .{ .n_tokens = 2, .grid_x = 2, .grid_y = 1, .video_chunk = 0 }, // vídeo B c0
    };
    var r = try expandVisionTokens(a, &prompt, &inputs, &.{0}, &.{ 1, 3 }, 999);
    defer r.deinit(a);
    try testing.expectEqual(@as(usize, 4), r.spans.len);
    // span0=still0 (marker img), span1-2=vídeo A chunks (marker idx1), span3=vídeo B (marker idx3)
    try testing.expectEqual(@as(usize, 0), r.spans[0].img);
    try testing.expectEqual(@as(usize, 1), r.spans[1].img);
    try testing.expectEqual(@as(usize, 2), r.spans[2].img);
    try testing.expectEqual(@as(usize, 3), r.spans[3].img);
    // pos_consumed: still 2 + vidA 2+2 + vidB 2 = 8
    try testing.expectEqual(@as(usize, 8), r.pos_consumed);
}

test "expandVisionTokens: t_factor vLLM (eje t avanza por chunk)" {
    const a = testing.allocator;
    const prompt = [_]u32{ 999, 1 }; // [vid_pad, texto]
    // 2 chunks, t_factor=2 (2s por par a 1 tps), grid 2x1
    const inputs = [_]ExpandInput{
        .{ .n_tokens = 2, .grid_x = 2, .grid_y = 1, .video_chunk = 0, .t_factor = 2.0 },
        .{ .n_tokens = 2, .grid_x = 2, .grid_y = 1, .video_chunk = 1, .t_factor = 2.0 },
    };
    var r = try expandVisionTokens(a, &prompt, &inputs, &.{}, &.{0}, 999);
    defer r.deinit(a);
    // chunk 0: pos_0=0, t=video_pos_0+0·2=0
    try testing.expectEqualSlices(i32, &.{ 0, 0, 0, 0 }, &r.pos_ids[0]);
    // chunk 1: pos_0=2, t=0+1·2=2 (≠pos_0: eje temporal vLLM)
    try testing.expectEqualSlices(i32, &.{ 2, 2, 2, 0 }, &r.pos_ids[2]);
    try testing.expectEqualSlices(i32, &.{ 2, 3, 2, 0 }, &r.pos_ids[3]);
}

test "expandVisionTokens: sin markers → sin expansión, pos lineal" {
    const a = testing.allocator;
    const prompt = [_]u32{ 1, 2, 3 };
    const inputs = [_]ExpandInput{.{ .n_tokens = 4, .grid_x = 2, .grid_y = 2 }};
    var r = try expandVisionTokens(a, &prompt, &inputs, &.{}, &.{}, 999);
    defer r.deinit(a);
    try testing.expectEqualSlices(u32, &prompt, r.ids);
    try testing.expectEqual(@as(usize, 0), r.spans.len);
    try testing.expectEqual(@as(usize, 0), r.pos_consumed);
    try testing.expectEqualSlices(i32, &.{ 2, 2, 2, 2 }, &r.pos_ids[2]);
}

test "expandVisionTokens: marker de vídeo agota chunks, imagen ignora chunks" {
    const a = testing.allocator;
    const prompt = [_]u32{ 999, 998, 1 };
    const inputs = [_]ExpandInput{
        .{ .n_tokens = 2, .grid_x = 2, .grid_y = 1, .video_chunk = 0 },
    };
    // marker img con SOLO chunks pendientes: no consume nada (tipo mismatch)
    var r = try expandVisionTokens(a, &prompt, &inputs, &.{0}, &.{}, 999);
    defer r.deinit(a);
    try testing.expectEqual(@as(usize, 0), r.spans.len);
    try testing.expectEqualSlices(u32, &prompt, r.ids);
}
