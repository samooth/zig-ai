# Qwen3.5 Hybrid — Gated DeltaNet (SSM) + Atención Completa

> Estado: 2026-08-20 — implementación completa (SSM con `QuantWeight`, bloque híbrido de
> atención + rutado por `isFullAttentionLayer`, validado contra llama.cpp con
> Qwen3.5-0.8B-Q4_0.gguf: Pearson 0.9989 en logits del primer token). Fase H ✅ en TODO.md.
> Fuentes autoritativas: `transformers/modular_qwen3_5.py` (HF), kernel de FLA
> `fla/ops/gated_delta_rule` (naive.py + fused_recurrent.py), y dump del GGUF real.

## 1. Arquitectura del modelo (confirmada contra el 9B real)

### 1.1 Topología del bloque híbrido

El `Qwen3_5DecoderLayer` es **pre-norm residual simple** (sin cross-branch):

```
residual = x
x = attn_norm(x)                      # input_layernorm → "attn_norm.weight"
x = ssm(x)  |  attention(x)           # según layer_types
x = residual + x
residual = x
x = post_attention_norm(x)            # "post_attention_norm.weight"
x = mlp(x)                            # SwiGLU (ffn_gate/up/down)
x = residual + x
```

### 1.2 Distribución de capas

Patrón 3:1 (`full_attention_interval` = 4, ausente del GGUF → default 4).
Capa `i` es atención completa si `(i+1) % 4 == 0`.

| Tipo | Capas (32 totales) |
|------|--------------------|
| SSM (Gated DeltaNet) | 0,1,2, 4,5,6, 8,9,10, …, 30 (24 capas) |
| Atención completa | 3, 7, 11, 15, 19, 23, 27, 31 (8 capas) |

`model_config.isFullAttentionLayer(il)` ya implementa esta regla.

### 1.3 Tensores por tipo de capa (dims del Qwen3.5-9B, n_embd=4096)

**Capa de atención completa (blk.3):**

| Tensor | dims | dtype | notas |
|--------|------|-------|-------|
| `attn_norm.weight` | [4096] | f32 | RMSNorm |
| `attn_q.weight` | [4096, 8192] | iq3_s | **fused Q+G**: 16 heads × 512 (Q 256 + G 256) |
| `attn_q_norm.weight` | [256] | f32 | RMSNorm por head (sobre 256 dims) |
| `attn_k.weight` | [4096, 1024] | iq3_s | 4 kv-heads × 256 |
| `attn_k_norm.weight` | [256] | f32 | RMSNorm por head |
| `attn_v.weight` | [4096, 1024] | q4_k | |
| `attn_output.weight` | [4096, 4096] | q4_k | GQA 16→4 (grupos de 4) |
| `post_attention_norm.weight` | [4096] | f32 | |

RoPE: `head_dim`(key_length)=256, `rope.dimension_count`=64 (solo 64 de 256 se
rotan), `rope.freq_base`=1e7, secciones IMROPE `[11, 11, 10, 0]`.

**Capa SSM (blk.0):**

| Tensor | dims | dtype | rol |
|--------|------|-------|-----|
| `attn_norm.weight` | [4096] | f32 | |
| `attn_qkv.weight` | [4096, 8192] | q4_k | Q,K,V fused (ver 2.1) |
| `attn_gate.weight` | [4096, 4096] | iq3_s | gate z (GatedRMSNorm) |
| `ssm_conv1d.weight` | [4, 8192] | f32 | depthwise causal conv, silu |
| `ssm_beta.weight` | [4096, 32] | iq3_s | update gate β |
| `ssm_alpha.weight` | [4096, 32] | iq3_s | decay gate a |
| `ssm_dt.bias` | [32] | f32 | |
| `ssm_a` | [32] | f32 | **A_log** (log-space, valores negativos ≈ −0.07) |
| `ssm_norm.weight` | [128] | f32 | GatedRMSNorm (por v-head) |
| `ssm_out.weight` | [4096, 4096] | iq3_s | proyección de salida |
| `post_attention_norm.weight` | [4096] | f32 | |

## 2. Gated DeltaNet (la recurrencia lineal)

### 2.1 Dims derivadas

```
key_dim   = d_state * n_group   = 128 * 16 = 2048   (Q, K)
value_dim = d_state * dt_rank   = 128 * 32 = 4096   (V, z) = d_inner
qkv_dim   = key_dim*2 + value_dim = 8192            (attn_qkv out)
n_k_heads = n_group             = 16
n_v_heads = dt_rank             = 32    (head_k_dim = head_v_dim = d_state = 128)
```

`ssm_norm` es la GatedRMSNorm: `y = rmsnorm(x, weight[128]) * silu(z)` aplicado
por v-head (dims 128 contiguas).

### 2.2 Recurrencia fiel al kernel de FLA

El kernel fused (`fused_recurrent_gated_delta_rule`, `USE_QK_L2NORM_IN_KERNEL`)
es la referencia exacta implementada en `src/transformer/ssm.zig`:

```
por token t, por v-head hv:
    hk = hv // (n_v_heads / n_k_heads)      # bloque (repeat_interleave)
    q = L2norm(q[hk]); k = L2norm(k[hk])    # /sqrt(sum+1e-6)
    g = -exp(A_log[hv]) * softplus(a[hv] + dt_bias[hv])
    S *= exp(g)
    d = beta[hv] * (v[hv] - S^T k)
    S += k ⊗ d
    o = S^T q * (1/sqrt(d_state))
```

`beta = sigmoid(ssm_beta @ X)`; `a = ssm_alpha @ X`.

### 2.3 Bugs corregidos (agosto 2026)

Dos desviaciones respecto al kernel de FLA fueron corregidas:

1. **Decay**: se usaba `S *= exp(softplus * ssm_a)` (decaimiento ~13× más débil).
   Ahora `S *= exp(-exp(ssm_a) * softplus)`. `ssm_a` es **A_log** (log-space): para
   el 9B los valores son ≈ −0.07 → `exp(A_log) ≈ 0.93` (decaimiento por paso < 1).
2. **Pairing de k-heads**: se usaba `hk = hv % n_group`; el kernel usa
   `i_h = i_hv // (HV // H)` (bloque). Corregido a `hk = hv // (n_v_heads / n_k_heads)`.
   No detectable con dims de test `n_group=1`; el test de recurrencia usa
   `n_k_heads=2, n_v_heads=4` (ratio 2) para ejercitarlo.

## 3. Estrategia de memoria: `QuantWeight`

`src/loader/quant_weight.zig` — referencia **zero-copy** a un tensor GGUF:

```zig
QuantWeight { info: *const gguf.TensorInfo, bytes: []const u8 }
```

- Sin descompresión en carga: los pesos cuantizados viven en el mmap.
- `dequantToF16(scratch)` / `dequantToF32(scratch)` descomprimen por bloques
  (≤256 elems de scratch) vía `gguf.dequantBlock`, volcando en el scratch f16
  persistente del layer; la proyección lineal lee directo del scratch.
- Pesos grandes de `ssm.zig` (`w_qkv`, `w_z`, `w_out`) usan `QuantWeight`.
- Pesos pequeños f32 se cargan en el tensor del layer vía `loadGgufF32Into`
  (rellena el buffer ya alocado, sin fugas).

## 4. Soporte de cuantización GGUF

`src/loader/gguf.zig` incluye `dequantBlock` (dispatch por bloque) y kernels
permanentes de tipos GGUF. Nuevos en esta fase: **IQ3_S** (test byte-exact vs C
permanente en el repo) e **IQ4_XS**, junto a IQ2_XS/IQ2_S. `dequantTensor` /
`dequantBlock` dan soporte para los tipos del 9B (iq3_s, q4_k, bf16, f32).

## 5. Estado de tests

- `zig build test`: **todos pasan** (55/55 con `GGUF_MODEL_PATH`; 52/55 sin él, 3 skips
  que requieren un `.gguf` real). Los tests de `tests/test_gguf.zig` se escribieron para el
  modelo 4B full-attention (esperan `attn_output`/`attn_q` en todas las capas); se ajustaron
  para el híbrido en la Fase H (`hybrid_attn.test.*` corregidos en `435f5b2`).
- Tests de `ssm.zig` (4): forward hand-computed, persistencia de estado,
  brute-force reference, y **recurrencia vs referencia naive de FLA**.
- Tests reales: `GGUF_MODEL_PATH=/opt/models/<archivo>.gguf zig build test`.

## 6. Próximos pasos (Fase H — bloque híbrido)

Fase H ✅ completada (ver TODO.md). Entregables: capa de atención completa con
`attn_q` fused Q+G, `attn_q_norm`/`attn_k_norm` por head, GQA 16→4, KV cache;
**IMROPE** en `src/transformer/rope.zig` (NEOX, rotar 64 de 256 dims, secciones
`[11, 11, 10, 0]`); rutado híbrido por `isFullAttentionLayer`; rework de
`src/loader/gguf_model.zig` a `QuantWeight`; CLI de inferencia validada contra el
modelo real (Pearson 0.9989).

## 7. Notas de entorno

- Repo en eCryptfs (sin `O_TMPFILE`): Zig 0.16 no puede crear archivos temp en
  el cache del repo. Usar `zig build test --cache-dir /tmp/opencode/zig-ai-cache`.
- El GGUF real está en `/opt/models/Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-IQ3_M.gguf`
  (arch `qwen35`, 32 capas, tokenizer gpt2, vocab 248320).
