// KVarN shared descriptor — contract surface between store/materialize and
// the native attention kernels. Transcribed for the zig-ai C1 record layout
// (fused K+V record per (head, group), 32-byte sector padding).
//
// Everything lives in the *rotated* WHT-128 domain: both the f16 stage and
// the packed records store post-WHT vectors. Consumers that need the original
// domain apply the (self-inverse) normalized butterfly themselves.
//
// Index array (i64 per token, one per stream):
//   enc >= 0   plain absolute cell, not explicitly staged
//   enc == -1  hole / skip
//   enc <= -2  explicitly staged: payload = u64(-(enc+2));
//              high 32 bits = stage slot + 1 (0 => no slot => treat as -1),
//              low 32 bits  = absolute cell
//
// Stage layout C2v2 (f16, one 128-wide row per slot/token/side):
//   stage[((stream * 128 * stage_groups + slot * 128 + pos) * 2*n_record_heads + (2h + side)) * 128 + dim]
// K rows live at 2*head, V rows at 2*head+1 — K and V are distinct
// projections of the same token, so they get separate stage rows. The
// caller passes record_head = 2*head (K side) or 2*head+1 (V side).
//
// Slot assignment:
//   non-SWA: group 0 -> slot 0 (resident sink); group g>0 -> 1+((g-1) % tail_groups)
//   SWA:     group g  -> g % stage_groups (absolute-tile ping-pong)

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#define KVARN_DIM 128
#define KVARN_STAGE_BYTES(slots, heads) \
    ((size_t)(slots) * (size_t)(heads) * KVARN_DIM * sizeof(__half))

// ---------------------------------------------------------------------------
// device-side descriptor
// ---------------------------------------------------------------------------

struct KvarnDesc {
    const uint8_t* records;      // packed C1 records (fused K+V per group/head)
    const __half*  stage;        // transient f16 staging (rotated domain)
    const int64_t* indices;      // per-token encoded positions (this stream)
    int n_record_heads;          // physical heads recorded per group
    int live_group;              // highest group with a valid token (this stream)
    int live_pos;                // position of the live token inside live_group
    int stream;                  // stream index this desc resolves
    int head_base;               // first physical head of the logical head
    int groups_per_stream;       // record ring depth per stream
    int record_bytes;            // bytes of one C1 record (layout.tile_bytes)
    int stage_groups;            // stage depth (>= 2)
    int tail_groups;             // hot (unsealed) groups kept in stage (>=1)
    int bits;                    // quant bits for THIS side (K or V)
    int value;                   // 0 = K side, 1 = V side
    int swa;                     // sliding-window ring mode
    int head_slices;             // 1/2/4 slices per logical head
    int head_dim;                // 64/128/256/512 — actual head dimension
    int eager_records;           // seal record as soon as a group completes
    int read_indirect;           // resolve cells through `indices`
    int original_domain;         // materialize/FA loader emits original domain
};

// ---------------------------------------------------------------------------
// index decoding
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t kvarn_index_payload(int64_t enc) {
    return enc < -1 ? (uint64_t)(-(enc + 2)) : (uint64_t)enc;
}

__device__ __forceinline__ int kvarn_index_stage_slot(int64_t enc) {
    const uint64_t payload = kvarn_index_payload(enc);
    const uint32_t packed = (uint32_t)(payload >> 32);
    return packed == 0 ? -1 : (int)(packed - 1u);
}

__device__ __forceinline__ int64_t kvarn_index_cell(int64_t enc) {
    return (int64_t)(uint32_t)kvarn_index_payload(enc);
}

// Resolves an encoded index into an absolute cell plus staging flags.
__device__ __forceinline__ int64_t kvarn_read_cell(
    int64_t enc, bool& explicitly_staged, int* assigned_slot)
{
    explicitly_staged = enc < -1;
    const uint64_t payload = kvarn_index_payload(enc);
    if (assigned_slot != nullptr) {
        const uint32_t packed = (uint32_t)(payload >> 32);
        *assigned_slot = packed == 0 ? -1 : (int)(packed - 1u);
    }
    return (int64_t)(uint32_t)payload;
}

// ---------------------------------------------------------------------------
// stage placement
// ---------------------------------------------------------------------------

__device__ __forceinline__ int kvarn_stage_slot_for_group(
    const KvarnDesc& d, int group, int assigned_slot)
{
    if (assigned_slot >= 0) return assigned_slot;
    if (d.swa) return group % d.stage_groups;
    return group == 0 ? 0 : 1 + ((group - 1) % d.tail_groups);
}

__device__ __forceinline__ int kvarn_stage_pos(
    const KvarnDesc& d, int group, int pos, int assigned_slot = -1)
{
    const int base = d.stream * KVARN_DIM * d.stage_groups;
    const int slot = kvarn_stage_slot_for_group(d, group, assigned_slot);
    return base + slot * KVARN_DIM + pos;
}

__device__ __forceinline__ const __half* kvarn_stage_row(
    const KvarnDesc& d, int stage_pos, int record_head)
{
    // C2v2: K/V rows interleaved — stride 2*n_record_heads. record_head
    // arrives as 2h (K) or 2h+1 (V) from the caller.
    return d.stage + ((int64_t)stage_pos * (2 * d.n_record_heads) + record_head) * KVARN_DIM;
}

// ---------------------------------------------------------------------------
// stage vs record membership
// ---------------------------------------------------------------------------

// True when `group` still lives in the f16 stage (hot window).
__device__ __forceinline__ bool kvarn_group_from_stage(const KvarnDesc& d, int group)
{
    if (d.eager_records) {
        if (!d.swa && group == 0) return true;  // sink stays resident
        return group == d.live_group
            && d.live_pos < KVARN_DIM - 1;
    }
    if (d.swa) {
        const int begin = d.live_group >= (d.tail_groups - 1)
            ? d.live_group - (d.tail_groups - 1) : 0;
        return group >= begin && group <= d.live_group;
    }
    return group == 0
        || (group > 0 && group <= d.live_group
            && group + (d.tail_groups - 1) >= d.live_group);
}

// True when `group` has been sealed into a compressed record and the record
// copy is authoritative (not shadowed by the stage).
__device__ __forceinline__ bool kvarn_group_from_record(const KvarnDesc& d, int group)
{
    if (group < 0 || kvarn_group_from_stage(d, group)) return false;
    if (d.eager_records) {
        const bool completed = group < d.live_group
            || (group == d.live_group && d.live_pos == KVARN_DIM - 1);
        if (!completed) return false;
        return d.swa
            ? d.live_group - group < d.groups_per_stream
            : group > 0 && group < d.groups_per_stream;
    }
    if (d.swa) {
        const int distance = d.live_group - group;
        return distance >= d.tail_groups
            && distance < d.groups_per_stream + d.tail_groups;
    }
    return group < d.live_group && group < d.groups_per_stream;
}

// ---------------------------------------------------------------------------
// C1 record addressing (fused K+V record per (group, head-slice))
// ---------------------------------------------------------------------------
// Layout C1 (from src/kv_cache/kvarn.zig, frozen):
//   [k_payload: packedBits(128*hd, kb)][k_s_col: hd f16][k_zp: hd f16]
//   [k_s_row: 128 f16]                                   <- indexed by TOKEN
//   [v_payload: packedBits(128*hd, vb)][v_s_col: hd f16][v_s_row: 128 f16][v_zp: 128 f16]  <pad 32B>
//
// K tile: row = dim, col = token  => per-row scale/zp absorb s_row(dim),
//         k_s_col[dim] = s_row(dim)*scale, k_zp[dim] = s_row(dim)*lo,
//         k_s_row[token] = s_col(token) (the "other" axis).
// V tile: row = token, col = dim  => v_s_row[token] = s_row(token)*scale,
//         v_zp[token] = s_row(token)*lo, v_s_col[dim] = s_col(dim) (other).
//
// Both tiles are ALWAYS 128x128 slices (B2 encodes per 128-dim slice, so a
// D>=256 logical head spans head_slices records — one per slice).

struct KvarnAxes {
    const uint8_t* payload;
    const __half* scale;   // tile ROW axis
    const __half* zp;      // tile ROW axis
    const __half* other;   // tile COLUMN axis
};

__device__ __forceinline__ const uint8_t* kvarn_record_base(
    const KvarnDesc& d, int record_group, int record_head)
{
    return d.records
        + ((int64_t)record_group * d.n_record_heads + record_head) * d.record_bytes;
}

// Axis pointers for the K side of a fused C1 record.
// hd = logical head_dim (region stride), row_axis_len = 128 tokens for
// k_s_row. All strides in BYTES.
__device__ __forceinline__ KvarnAxes kvarn_k_axes(
    const uint8_t* rec, int payload_bytes, int hd)
{
    KvarnAxes ax;
    ax.payload = rec;
    ax.scale = (const __half*)(rec + payload_bytes);            // k_s_col[hd]
    ax.zp = (const __half*)(rec + payload_bytes + hd * 2);      // k_zp[hd]
    ax.other = (const __half*)(rec + payload_bytes + hd * 4);    // k_s_row[128]
    return ax;
}

// Axis pointers for the V side. `off_v` = byte offset of the V region within
// the fused record (layout.v_payload_off).
__device__ __forceinline__ KvarnAxes kvarn_v_axes(
    const uint8_t* rec, int off_v, int payload_bytes, int hd)
{
    // Layout C1 real (kvarn.zig tileLayout): tras v_payload viene
    // [v_s_col hd][v_s_row 128][v_zp 128] — NO [v_s_row][v_zp][v_s_col].
    // La versión anterior permutaba los 3 ejes (scale leia v_s_col,
    // zp leia v_s_row, other leia v_zp) ⇒ V records desviado ~170x.
    // Dequant V correcto: (q·v_s_row[pos] + v_zp[pos])·v_s_col[dim].
    KvarnAxes ax;
    const uint8_t* v = rec + off_v;
    ax.payload = v;
    ax.scale = (const __half*)(v + payload_bytes + hd * 2);        // v_s_row[128]
    ax.zp = (const __half*)(v + payload_bytes + hd * 2 + 128 * 2); // v_zp[128]
    ax.other = (const __half*)(v + payload_bytes);                 // v_s_col[hd]
    return ax;
}

// Row-major bit unpack, LSB-first stream (matches B2 packBit).
// Fast paths for 8/4/2 bits; generic window for 3/5/6.
__device__ __forceinline__ uint32_t kvarn_unpack(
    const uint8_t* payload, int index, int bits)
{
    if (bits == 8) return payload[index];
    if (bits == 4) {
        const uint8_t packed = payload[index >> 1];
        return (packed >> ((index & 1) * 4)) & 0x0fu;
    }
    if (bits == 2) {
        const uint8_t packed = payload[index >> 2];
        return (packed >> ((index & 3) * 2)) & 0x03u;
    }
    const int bit_off = index * bits;
    const int byte_off = bit_off >> 3;
    const int shift = bit_off & 7;
    uint32_t window = payload[byte_off];
    if (shift + bits > 8) window |= (uint32_t)payload[byte_off + 1] << 8;
    return (window >> shift) & ((1u << bits) - 1u);
}

// Dual unpack for two consecutive elements at an even index: one window and
// one shift computation serve both output lanes (MMA loaders consume
// dimensions in pairs).
__device__ __forceinline__ uint32_t kvarn_unpack_pair(
    const uint8_t* payload, int index, int bits)
{
    if (bits == 8) {
        return (uint32_t)payload[index] | ((uint32_t)payload[index + 1] << 8);
    }
    if (bits == 4) {
        const uint8_t packed = payload[index >> 1];
        return (packed & 0x0fu) | ((uint32_t)(packed >> 4) << 8);
    }
    if (bits == 2) {
        const uint8_t packed = payload[index >> 2];
        const int shift = (index & 3) * 2;
        return ((packed >> shift) & 0x03u)
            | (((uint32_t)((packed >> (shift + 2)) & 0x03u)) << 8);
    }
    const int bit_off = index * bits;
    const int byte_off = bit_off >> 3;
    const int shift = bit_off & 7;
    const uint32_t window =
        (uint32_t)payload[byte_off] | ((uint32_t)payload[byte_off + 1] << 8);
    const uint32_t mask = (1u << bits) - 1u;
    return ((window >> shift) & mask)
        | ((((window >> (shift + bits)) & mask)) << 8);
}

// Resolve one rotated-domain value from a record tile.
//   token/pos: position inside the 128-token group
//   dim:       local dimension inside the 128-dim slice
//   slice:     which 128-dim slice of the logical head (D>=256)
// `off_v` = v_payload_off of the layout; both sides' bits come from the desc.
__device__ __forceinline__ float kvarn_record_value(
    const KvarnDesc& d, int record_group, int token_pos, int dim, int slice)
{
    const int record_head = d.head_base + slice;
    const uint8_t* rec = kvarn_record_base(d, record_group, record_head);
    const int hd = d.head_dim; // slice-local dim; hd region stride (64/128/256/512)
    const int k_payload_bytes = KVARN_DIM * hd * d.bits / 8;

    if (!d.value) {
        const KvarnAxes ax = kvarn_k_axes(rec, k_payload_bytes, hd);
        const uint32_t q = kvarn_unpack(ax.payload, dim * KVARN_DIM + token_pos, d.bits);
        return ((float)q * __half2float(ax.scale[dim])
                   + __half2float(ax.zp[dim]))
            * __half2float(ax.other[token_pos]);
    }
    // V side: need v_payload_off; recomputed from K region size.
    const int k_region = k_payload_bytes + hd * 2 + hd * 2 + KVARN_DIM * 2;
    const int off_v = k_region;
    const int v_payload_bytes = KVARN_DIM * hd * d.bits / 8;
    const KvarnAxes ax = kvarn_v_axes(rec, off_v, v_payload_bytes, hd);
    const uint32_t q = kvarn_unpack(ax.payload, token_pos * KVARN_DIM + dim, d.bits);
    return ((float)q * __half2float(ax.scale[token_pos])
               + __half2float(ax.zp[token_pos]))
        * __half2float(ax.other[dim]);
}
// (guard: #pragma once al inicio del fichero)