# Qwen3.8 (Unsloth) TODO

> **Objetivo**: Soporte del modelo **Qwen3.8-27B** (arquitectura `qwen35`, GGUFs de
> Unsloth Dynamic) incluyendo las cuantizaciones de **1 bit**. El AirLLM layer
> streaming ya está implementado, por lo que se pueden cargar modelos mayores que
> la VRAM (7.7 GiB en RTX 3080 Laptop sm_86).

---

## 📋 Estado Actual

| Componente | Estado |
|---|---|
| Arquitectura `qwen35` (híbrido SSM + attention) | ✅ Soportada (Qwen3.5-0.8B, `hybrid_layer.zig`) |
| AirLLM layer streaming | ✅ Implementado (modelos > VRAM) |
| Cuantización 1-bit IQ1_S / IQ1_M / IQ2_* / IQ3_* / IQ4_XS | ✅ Ya soportada (CPU + CUDA) |
| Qwen3.8-27B carga/verificación | ⏳ Pendiente (descargar GGUF) |
| `attention.head_count_kv` escalar en qwen35 | ⚠️ Verificar (ver Fase 2) |
| Tipos 1-bit del fork `iq1-narrow` (Q1_0/Q2_0/IQ1_XS/XXS/XXXS) | ❌ No soportados (solo 2.4T) |

---

## 🎯 Datos del modelo (config.json → `unsloth/Qwen3.8-27B-GGUF`)

- `model_type`: `qwen3_5` / `qwen3_5_text` → arch GGUF **`qwen35`** (ya soportada).
- **64 capas**, `full_attention_interval: 4` → 16 full attention + 48 linear attention
  (Gated DeltaNet, SSM).
- `hidden_size`: 5120, `intermediate_size`: 17408.
- Attention full: `num_attention_heads: 24`, **`num_key_value_heads: 4`**,
  `head_dim: 256`, `attn_output_gate: true`, `output_gate_type: swish`.
- Linear/SSM: `linear_conv_kernel_dim: 4`, `linear_key_head_dim: 128`,
  `linear_num_key_heads: 16`, `linear_num_value_heads: 48`,
  `linear_value_head_dim: 128`, `mamba_ssm_dtype: float32`.
- RoPE: `partial_rotary_factor: 0.25`, `rope_theta: 1e7`,
  `mrope_section: [11, 11, 10]` (interleaved).
- `vocab_size: 248320`, `max_position_embeddings: 262144`,
  `tie_word_embeddings: false`.
- MTP: `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`,
  `unsloth_fixed_mtp: true`.
- Tokens especiales: bos/eos `248044`, pad `248055`, image `248056`,
  video `248057`, vision_start `248053`, vision_end `248054`.
- Vision: `mmproj-*.gguf` (fuera de alcance para inferencia de texto).

---

## 📦 Repo GGUF (unsloth/Qwen3.8-27B-GGUF)

Todos los archivos usan **tipos GGUF ya soportados** por el engine (la columna
"Tipo" es el dtype GGUF real, no el nombre del archivo):

| Archivo | Tamaño | Tipo GGUF | Soportado |
|---|---|---|---|
| `Qwen3.8-27B-UD-IQ1_M.gguf` | 6.73 GB | IQ1_M (29) | ✅ |
| `Qwen3.8-27B-UD-IQ1_S.gguf` | 6.19 GB | IQ1_S (19) | ✅ |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | 7.27 GB | IQ2_XXS | ✅ |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | 8.37 GB | IQ2_S | ✅ |
| `Qwen3.8-27B-UD-IQ3_XXS.gguf` | 10.9 GB | IQ3_XXS | ✅ |
| `Qwen3.8-27B-UD-IQ3_S.gguf` | 12 GB | IQ3_S | ✅ |
| `Qwen3.8-27B-UD-IQ4_XS.gguf` | 14.3 GB | IQ4_XS | ✅ |
| `Qwen3.8-27B-UD-Q4_K_M/S/XL`, `UD-Q5_K_*`, `UD-Q6_K_*`, `UD-Q8_K_*` | ~15-30 GB | K-quants | ✅ |
| `Qwen3.8-27B-Q4_0.gguf` / `Q4_1` / `Q8_0` | 16.1 / 17.5 / 29 GB | Q4_0/Q4_1/Q8_0 | ✅ |
| `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | — | MTP draft separado | — |
| `mmproj-BF16.gguf`, `mmproj-F16.gguf` | — | vision | fuera de alcance |

> Modelo objetivo para pruebas: **`Qwen3.8-27B-UD-IQ1_S.gguf` (6.19 GB)** — 1 bit,
> cabe en VRAM, tipo ya soportado. Alternativas mayores vía layer streaming.

---

## 🆕 Tipos 1-bit del fork `iq1-narrow` (solo Qwen3.8-2.4T, 397 GB)

El 27B **no** usa estos tipos; el modelo 2.4T-A95B (MoE, 95B activos) sí. Para
completitud de "1-bit" se documentan aquí. Fork: `unslothai/llama.cpp` branch
`iq1-narrow` (PR #61; seguimiento en `mudler/vllm.cpp#933`).

| Tipo | Valor | bpw | QK | Block bytes | Index/Grid |
|---|---|---|---|---|---|
| `NVFP4` | 40 | — | — | — | Blackwell-only (no aplica sm_86) |
| `Q1_0` | 41 | 1.0 | 128 | 18 | sign-bit solo, `d = sum_abs/qk` |
| `Q2_0` | 42 | 2.0 | 64 | 18 | 2-bit |
| `IQ1_XS` | 64 | 1.4375 | 256 | 46 | 10-bit, grid 1024 (HF name "TQ2_0") |
| `IQ1_XXS` | 65 | 1.3125 | 256 | 42 | 9-bit, grid 512 (HF name "TQ1_0") |
| `IQ1_XXXS` | 66 | 1.1875 | 256 | 38 | 8-bit, grid 256 (HF name "Q1_0") |

Notas:
- Los nombres de archivo HF (`TQ2_0`/`TQ1_0`/`Q1_0`) **no** son el dtype GGUF real.
- `IQ1_XS/XXS/XXXS` reutilizan el codebook ternario 8-dim de IQ1_S con índice más
  angosto → dequant ≈ `dequantIq1_s` (`gguf.zig:1890`), reusar `iq1s_grid`
  (`gguf.zig:1032-1359`).
- Los tipos fork-locales empiezan en 64 (42..63 reservados para upstream).

---

## 🔧 Gaps detectados en el engine

1. **`attention.head_count_kv` escalar en qwen35** — `model_config.fromGguf` solo
   lee `attention.head_count`; para qwen35 `head_count_kv` **queda = head_count**
   (`model_config.zig:60`). Qwen3.8-27B tiene **4 KV heads vs 24 Q heads**.
   - Verificar si el GGUF escribe `qwen35.attention.head_count_kv` (escalar).
   - Verificar qué usa Qwen3.5-0.8B (modelo de test) actualmente.
   - Fix: leer escalar en el path qwen35 (como LFM2 pero sin array per-layer).
   - El path GQA ya existe (`gqa.zig`) y `hybrid_attn` usa `n_kv_head`
     (`hybrid_layer.zig:69`) → correcto si `head_count_kv` se lee bien.
2. **Tipos GGUF nuevos** (40/41/42/64/65/66) — faltan en `GgmlType`
   (`gguf.zig:25-130`), `blockSize`, `dequantBlock` (`gguf.zig:2168-2213`),
   kernels CUDA (`/kernels/*.cu` + `launcherFor` en `gguf_dequant_gpu.zig`).
3. **`dequantTensor` restrictivo** (`gguf.zig:2223-2250`) — rechaza IQ1_S/IQ2_S/
   IQ1_M/tq1_0/tq2_0/mxfp4 y por extensión los nuevos tipos; usado para tensores
   raíz (embedding/output_norm) en `gguf_model.zig:74,104`, `hybrid_layer.zig:695`,
   `ssm.zig:723`, `layer.zig:627,655`. Revisar para 1-bit.
4. **MTP**: detección `draft-mtp` vía `blk.0.nextn.eh_proj.weight`
   (`main.zig:612-631`); driver especulativo aún "pendiente". Verificar si el GGUF
   principal embebe tensores `nextn.*` y que el loader los ignore (lookup por
   nombre → sí, `getTensor`).
5. **Fused quant GEMM** solo `q4_0/q4_1/q5_k/q6_k` — 1-bit cae a host dequant +
   SGEMM (aceptable para capa streaming).

---

## 📋 TODO Detallado

## 🟢 FASE 1: Cargar y validar Qwen3.8-27B-UD-IQ1_S (6.19 GB)

- [ ] Descargar `Qwen3.8-27B-UD-IQ1_S.gguf` a `/opt/models/`.
- [ ] Volcar metadata GGUF: claves `qwen35.*` (block_count, attention.key_length,
      full_attention_interval, head_count_kv, ssm.*, rope.dimension_sections) y
      comparar contra `model_config.zig` (FASE 1.1).
- [ ] Verificar nombres de tensores: `blk.{i}.attn_*`, `blk.{i}.ssm_*`,
      `blk.{i}.ffn_*`, `token_embd.weight`, `output_norm.weight`,
      `output.weight`, y presencia/ausencia de `nextn.*` (MTP embebido).
- [ ] Confirmar que el loader ignora tensores no solicitados (MTP embebido).
- [ ] Ejecutar `zig build test` con `GGUF_MODEL_PATH` apuntando al modelo →
      E1/E2 (carga + forward híbrido) y validar logits contra llama.cpp.

## 🟡 FASE 2: Fix `head_count_kv` escalar para qwen35

- [ ] Verificar `qwen35.attention.head_count_kv` en el GGUF del 27B y del 0.8B.
- [ ] Leer el escalar en el path qwen35 de `model_config.fromGguf`
      (`u64Meta(..., "attention.head_count_kv", head_count)`).
- [ ] Confirmar GQA 24→4 en `hybrid_attn` / `paged_attention`
      (`.num_kv_heads = cfg.head_count_kv` en `main.zig:638,675`).
- [ ] Test unitario: `model_config` con metadata qwen35 head_count_kv=4.

## 🟠 FASE 3: Tipos 1-bit del fork `iq1-narrow` (opcional, solo 2.4T)

- [ ] Añadir `GgmlType` 40 (`nvfp4`), 41 (`q1_0`), 42 (`q2_0`), 64 (`iq1_xs`),
      65 (`iq1_xxs`), 66 (`iq1_xxxs`) en `gguf.zig:25-130`.
- [ ] `blockSize` + `nameToType`/parse para los nuevos tipos.
- [ ] Dequant CPU:
      - `Q1_0`/`Q2_0`: sign-bit / 2-bit simples (`quantize_row_q1_0_ref` del fork:
        QK 128, 18 B, `d = sum_abs/qk`).
      - `IQ1_XS/XXS/XXXS`: reusar `iq1s_grid` con indexado narrowing (10/9/8 bit).
- [ ] Kernels CUDA (`/kernels/dequant_*.cu`) + wiring `launcherFor`
      (`gguf_dequant_gpu.zig`).
- [ ] Revisar restricciones de `dequantTensor` (`gguf.zig:2223-2250`).
- [ ] Tests round-trip quantize/dequantize + validación contra el fork.

## 🟣 FASE 4: Integración y validación final

- [ ] Probar 27B a mayor precisión (p.ej. `UD-Q4_K_M` ~15 GB) vía layer streaming
      (modelo > VRAM).
- [ ] Benchmark 1-bit IQ1_S: tokens/s CPU vs GPU, pico VRAM.
- [ ] (Opcional) Decodificación especulativa MTP con el draft separado.

---

## 📝 Notas de Implementación

- `GgmlType` enum: `src/loader/gguf.zig:25-130` (hasta `mxfp4`=39; `tq1_0`=34,
  `tq2_0`=35). Faltan 40/41/42/64/65/66.
- Dequant CPU dispatch único: `dequantBlock` (`gguf.zig:2168-2213`).
- Kernels CUDA en **`/kernels/*.cu`** (16 archivos, no `src/kernels/`), cableados
  por `launcherFor` en `src/loader/gguf_dequant_gpu.zig`.
- `QuantWeight` (`src/loader/quant_weight.zig`): cero-copia mmap + dequant lazy.
- Añadir nuevas entradas al checklist de `docs/README.md` (índice central).

---

## 🔗 Referencias

- Modelo: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF (config.json embebido)
- Docs Unsloth: https://unsloth.ai/docs/models/qwen3.8
- Fork llama.cpp `iq1-narrow`: https://github.com/unslothai/llama.cpp/tree/iq1-narrow
  - `ggml/include/ggml.h` (enum tipos 40/41/42/64/65/66)
  - `ggml/src/ggml-common.h` (layouts)
  - `ggml/src/ggml-quants.c` (`quantize_row_q1_0_ref`, `dequantize_row_*`)
- Seguimiento soporte externo: https://github.com/mudler/vllm.cpp/issues/933
- Arquitectura local: `src/loader/gguf.zig`, `src/loader/model_config.zig`,
  `src/loader/gguf_dequant_gpu.zig`, `/kernels/*.cu`, `src/transformer/hybrid_*`