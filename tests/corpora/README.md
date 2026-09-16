# Corpora de tests PPL (lane-f F1v7)

Corpora de texto para harnesses/tests de perplexity. Deterministas
(versionados en el repo) — /tmp fue barrido 2 veces en una sola sesión
y mató runs en silencio (test moría en el open del corpus).

## ppl_sentences96.txt (549B)
15 frases simples repetibles ("The capital of France is Paris." ×5 +
10 frases triviales). Tokeniza a ~96-116 tokens según política BOS.
Diseñado para diagnóstico por-posición (el patrón "capital of X is Y"
repetido expone la retención de sujeto a distancia creciente).

## ppl_wiki12k.txt (12KB)
Primeros 12000 bytes de texto wikitext natural (fuente: wikitext raw
de la instalación de exllamav3 en /ai — primera sección, ~2959 tokens
con el tokenizer Llama-3.2). Es el corpus del gate E2E golden:

- Golden llama.cpp (llama-perplexity -c 1024 --chunks 1):
  - Q3_K_M: PPL 6.9635 ± 0.81
  - Q8_0:   PPL 6.4131 ± 0.73
- zig-ai engine legacy (post F1v7 RoPE NORM, --ppl --ctx-size 2048):
  - Q3_K_M: PPL 12.66 (residual 1.8× bajo investigación)
  - Q8_0:   en vuelo

Uso:
```bash
# zig-ai
./zig-out/bin/zig-ai-engine --model <gguf> \
  --ppl tests/corpora/ppl_wiki12k.txt --backend cpu --ctx-size 2048

# golden llama.cpp (misma semántica de ventana aprox)
llama-perplexity -m <gguf> -f tests/corpora/ppl_wiki12k.txt --chunks 1 -c 1024
```

Nota: con `-c 1024` llama.cpp divide en 2 secuencias de ~1479 tokens;
el motor usa window 2048/stride 1024 (2 chunks) — la historia efectiva
por token difiere ligeramente (motor da MÁS contexto ⇒ PPL esperado
ligeramente mejor, no peor; comparar con ese sesgo en mente).

## Fixtures .ktb (KT-B lane-f, 2026-09-12)

Pesos del runtime KV-transfer (`--kv-transfer <path>`, formato ZKTB v1 —
véase `src/kv_cache/kt_transfer.zig`). Regenerables (no commiteados,
`.gitignore *.ktb`):

- `ktb_gate0.ktb` — identity 2 capas × 3 kv × hd 64 (test unitario gate-0).
- `ktb_dense_I.ktb` — dense W=I numérico 2×2×16 (regresión fromSlice).
- `ktb_err.ktb` — identity 1×2×32 (error paths).
- `ktb_llama1b_identity.ktb` — identity 16 capas × 8 kv × hd 64 (CLI E2E
  Llama-3.2-1B; regenerar: `writeIdentityKtb(io, path, 16, 8, 64)`).
- `ktb_llama1b_denseI.ktb` — dense W=I 16×8×64 (CLI E2E path gemm; genera
  con numpy: I(512×512) por capa + bias 0).

Gate CLI (los 3 deben dar ' Paris.' idéntico, RC=0):
```bash
M=/ai/models/Llama-3.2-1B-Instruct-Q3_K_M.gguf
./zig-out/bin/zig-ai-engine --model $M --prompt "The capital of France is" \
  -n 4 --backend cpu --ctx-size 1024                      # baseline
./zig-out/bin/zig-ai-engine --model $M ... --kv-transfer tests/corpora/ktb_llama1b_identity.ktb
./zig-out/bin/zig-ai-engine --model $M ... --kv-transfer tests/corpora/ktb_llama1b_denseI.ktb
```
OJO: `--ctx-size` explícito — ctx default 65536 OOMs el pool legacy (256MB),
y `--ctx` (sin -size) NO existe: se engulle en silencio.
